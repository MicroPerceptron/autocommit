use std::io::Write;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use autocommit_core::CoreError;
use gix::bstr::{BStr, BString, ByteSlice};

pub(crate) struct Repo {
    inner: gix::Repository,
}

impl Repo {
    pub(crate) fn discover() -> Result<Self, CoreError> {
        let cwd = std::env::current_dir()
            .map_err(|err| CoreError::Io(format!("failed to read current directory: {err}")))?;
        let inner = gix::discover_with_environment_overrides(&cwd)
            .map_err(|err| CoreError::Io(format!("failed to discover git repository: {err}")))?;
        Ok(Self { inner })
    }

    pub(crate) fn diff_cached(&self) -> Result<String, CoreError> {
        let index = self
            .inner
            .index_or_empty()
            .map_err(|err| CoreError::Io(format!("failed to read index: {err}")))?;
        let head_tree_id = self
            .inner
            .head_tree_id_or_empty()
            .map_err(|err| CoreError::Io(format!("failed to resolve HEAD tree: {err}")))?
            .detach();

        let mut out = String::new();
        self.inner
            .tree_index_status(
                head_tree_id.as_ref(),
                &index,
                None,
                gix::status::tree_index::TrackRenames::Disabled,
                |change, _, _| {
                    append_tree_index_patch(&self.inner, &mut out, change)?;
                    Ok::<_, CoreError>(ControlFlow::Continue(()))
                },
            )
            .map_err(|err| CoreError::Io(format!("failed to compute staged diff: {err}")))?;

        Ok(out)
    }

    pub(crate) fn diff_worktree(&self) -> Result<String, CoreError> {
        let (mut pipeline, index) = self
            .inner
            .filter_pipeline(None)
            .map_err(|err| CoreError::Io(format!("failed to configure filter pipeline: {err}")))?;
        let index_state: &gix::index::State = &index;

        let mut out = String::new();
        let mut iter = self
            .inner
            .status(gix::progress::Discard)
            .map_err(|err| CoreError::Io(format!("failed to initialize status: {err}")))?
            .index(index.clone())
            .index_worktree_rewrites(None)
            .untracked_files(gix::status::UntrackedFiles::Files)
            .into_index_worktree_iter(Vec::<BString>::new())
            .map_err(|err| CoreError::Io(format!("failed to inspect worktree status: {err}")))?;

        for item in &mut iter {
            let item = item
                .map_err(|err| CoreError::Io(format!("failed to read worktree status: {err}")))?;
            append_index_worktree_patch(&self.inner, &mut pipeline, index_state, &mut out, item)?;
        }

        if let Some(mut outcome) = iter.into_outcome() {
            if let Some(result) = outcome.write_changes() {
                result.map_err(|err| {
                    CoreError::Io(format!("failed to update index metadata: {err}"))
                })?;
            }
        }

        Ok(out)
    }

    pub(crate) fn repo_root(&self) -> PathBuf {
        self.inner
            .workdir()
            .map(PathBuf::from)
            .unwrap_or_else(|| self.inner.git_dir().to_path_buf())
    }

    pub(crate) fn common_git_dir(&self) -> &Path {
        self.inner.common_dir()
    }

    pub(crate) fn commit(
        &self,
        message: &str,
        staged_only: bool,
        _no_verify: bool,
        sign_commit: bool,
    ) -> Result<(), CoreError> {
        let tree_id = if staged_only {
            self.tree_id_for_index()?
        } else {
            self.tree_id_for_worktree()?
        };

        let parent_id = self
            .inner
            .head()
            .map_err(|err| CoreError::Io(format!("failed to resolve HEAD: {err}")))?
            .id()
            .map(|id| id.detach());

        if sign_commit {
            let commit_id = self.create_signed_commit(&tree_id, parent_id.as_ref(), message)?;
            self.update_head_ref(&commit_id, parent_id.as_ref())?;
        } else {
            let parents = parent_id.iter().cloned().collect::<Vec<_>>();
            self.inner
                .commit("HEAD", message, tree_id, parents)
                .map_err(|err| CoreError::Io(format!("failed to create commit: {err}")))?;
        }

        let mut index = self.inner.index_from_tree(&tree_id).map_err(|err| {
            CoreError::Io(format!("failed to rebuild index from commit tree: {err}"))
        })?;
        index
            .write(gix::index::write::Options {
                extensions: Default::default(),
                skip_hash: false,
            })
            .map_err(|err| CoreError::Io(format!("failed to write index: {err}")))?;

        Ok(())
    }

    pub(crate) fn signoff_identity(&self) -> Result<String, CoreError> {
        let name = self.git_config_value("user.name")?;
        let email = self.git_config_value("user.email")?;
        if name.is_empty() || email.is_empty() {
            return Err(CoreError::Io(
                "git `user.name` and `user.email` are required for signoff".to_string(),
            ));
        }
        Ok(format!("{name} <{email}>"))
    }

    pub(crate) fn signing_key(&self) -> Result<Option<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["config", "--get", "user.signingkey"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git config: {err}")))?;
        if !output.status.success() {
            return Ok(None);
        }

        let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if key.is_empty() {
            Ok(None)
        } else {
            Ok(Some(key))
        }
    }

    pub(crate) fn set_signing_key(&self, key: &str, global: bool) -> Result<(), CoreError> {
        let repo_root = self.repo_root();
        let mut command = Command::new("git");
        command.arg("config");
        if global {
            command.arg("--global");
        } else {
            command.arg("--local");
        }
        let output = command
            .args(["user.signingkey", key])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git config: {err}")))?;
        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git config user.signingkey failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(())
    }

    pub(crate) fn push(&self) -> Result<(), CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .arg("push")
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git push: {err}")))?;

        if output.status.success() {
            return Ok(());
        }

        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let mut detail = String::new();
        if !stderr.is_empty() {
            detail.push_str(&stderr);
        }
        if !stdout.is_empty() {
            if !detail.is_empty() {
                detail.push_str("; ");
            }
            detail.push_str(&stdout);
        }
        if detail.is_empty() {
            detail.push_str("unknown error");
        }

        Err(CoreError::Io(format!("git push failed: {detail}")))
    }

    pub(crate) fn current_branch(&self) -> Result<Option<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git rev-parse: {err}")))?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git rev-parse failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if name.is_empty() || name == "HEAD" {
            Ok(None)
        } else {
            Ok(Some(name))
        }
    }

    pub(crate) fn local_branches(&self) -> Result<Vec<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["for-each-ref", "refs/heads", "--format=%(refname:short)"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git for-each-ref: {err}")))?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git for-each-ref failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut branches = stdout
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        branches.sort();
        Ok(branches)
    }

    pub(crate) fn remote_branches(&self) -> Result<Vec<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["for-each-ref", "refs/remotes", "--format=%(refname:short)"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git for-each-ref: {err}")))?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git for-each-ref failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut branches = stdout
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .filter(|line| !line.ends_with("/HEAD"))
            .collect::<Vec<_>>();
        branches.sort();
        Ok(branches)
    }

    pub(crate) fn remote_default_branch(&self) -> Result<Option<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["symbolic-ref", "refs/remotes/origin/HEAD"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git symbolic-ref: {err}")))?;

        if !output.status.success() {
            return Ok(None);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let value = stdout.trim();
        if value.is_empty() {
            return Ok(None);
        }
        let name = value.trim_start_matches("refs/remotes/").to_string();
        if name.is_empty() {
            Ok(None)
        } else {
            Ok(Some(name))
        }
    }

    pub(crate) fn remote_names(&self) -> Result<Vec<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .arg("remote")
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git remote: {err}")))?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git remote failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut remotes = stdout
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();
        remotes.sort();
        Ok(remotes)
    }

    pub(crate) fn remote_branch_exists(
        &self,
        remote: &str,
        branch: &str,
    ) -> Result<bool, CoreError> {
        let repo_root = self.repo_root();
        let ref_name = format!("refs/remotes/{remote}/{branch}");
        let output = Command::new("git")
            .args(["show-ref", "--verify", "--quiet", &ref_name])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git show-ref: {err}")))?;

        Ok(output.status.success())
    }

    pub(crate) fn diff_range(&self, base: &str, head: &str) -> Result<String, CoreError> {
        let repo_root = self.repo_root();
        let range = format!("{base}...{head}");
        let output = Command::new("git")
            .args(["diff", "--no-color", &range])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git diff: {err}")))?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git diff failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    fn create_signed_commit(
        &self,
        tree_id: &gix::hash::ObjectId,
        parent_id: Option<&gix::hash::ObjectId>,
        message: &str,
    ) -> Result<String, CoreError> {
        let repo_root = self.repo_root();
        let mut command = Command::new("git");
        command
            .arg("commit-tree")
            .arg(tree_id.to_string())
            .arg("-S");
        if let Some(parent_id) = parent_id {
            command.arg("-p").arg(parent_id.to_string());
        }
        command
            .args(["-F", "-"])
            .current_dir(&repo_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command
            .spawn()
            .map_err(|err| CoreError::Io(format!("failed to run git commit-tree: {err}")))?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(message.as_bytes()).map_err(|err| {
                CoreError::Io(format!(
                    "failed to write commit message to git commit-tree: {err}"
                ))
            })?;
        }
        let output = child.wait_with_output().map_err(|err| {
            CoreError::Io(format!("failed to read git commit-tree output: {err}"))
        })?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if signing_tool_missing(&stderr) {
                return Err(CoreError::Io(
                    "git commit signing requires `gpg`, but it is not available. Install `gpg` or disable signing with `autocommit commit --configure-commit-policy`".to_string(),
                ));
            }
            if signing_secret_key_missing(&stderr) {
                return Err(CoreError::Io(
                    "git commit signing failed: no usable GPG secret key. Run `gpg --full-generate-key` (or import a key), then set `git config --global user.signingkey <KEYID>`.".to_string(),
                ));
            }
            return Err(CoreError::Io(format!("git commit-tree failed: {}", stderr)));
        }

        let commit_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if commit_id.is_empty() {
            return Err(CoreError::Io(
                "git commit-tree returned an empty commit id".to_string(),
            ));
        }
        Ok(commit_id)
    }

    fn update_head_ref(
        &self,
        commit_id: &str,
        parent_id: Option<&gix::hash::ObjectId>,
    ) -> Result<(), CoreError> {
        let repo_root = self.repo_root();
        let target_ref = self
            .head_target_ref()?
            .unwrap_or_else(|| "HEAD".to_string());
        let mut command = Command::new("git");
        command
            .arg("update-ref")
            .arg(&target_ref)
            .arg(commit_id)
            .current_dir(&repo_root);
        if let Some(parent_id) = parent_id {
            command.arg(parent_id.to_string());
        }
        let output = command
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git update-ref: {err}")))?;
        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git update-ref failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(())
    }

    fn head_target_ref(&self) -> Result<Option<String>, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["symbolic-ref", "-q", "HEAD"])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git symbolic-ref: {err}")))?;
        if !output.status.success() {
            return Ok(None);
        }

        let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if value.is_empty() {
            Ok(None)
        } else {
            Ok(Some(value))
        }
    }

    fn git_config_value(&self, key: &str) -> Result<String, CoreError> {
        let repo_root = self.repo_root();
        let output = Command::new("git")
            .args(["config", "--get", key])
            .current_dir(&repo_root)
            .output()
            .map_err(|err| CoreError::Io(format!("failed to run git config: {err}")))?;
        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "git config --get {key} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn tree_id_for_index(&self) -> Result<gix::hash::ObjectId, CoreError> {
        let index = self
            .inner
            .index_or_empty()
            .map_err(|err| CoreError::Io(format!("failed to read index: {err}")))?;
        let head_tree_id = self
            .inner
            .head_tree_id_or_empty()
            .map_err(|err| CoreError::Io(format!("failed to resolve HEAD tree: {err}")))?
            .detach();

        let mut editor = self
            .inner
            .edit_tree(head_tree_id)
            .map_err(|err| CoreError::Io(format!("failed to initialize tree editor: {err}")))?;

        self.inner
            .tree_index_status(
                head_tree_id.as_ref(),
                &index,
                None,
                gix::status::tree_index::TrackRenames::Disabled,
                |change, _, _| {
                    apply_tree_index_change(&mut editor, change)?;
                    Ok::<_, CoreError>(ControlFlow::Continue(()))
                },
            )
            .map_err(|err| CoreError::Io(format!("failed to compute tree/index delta: {err}")))?;

        editor
            .write()
            .map(|id| id.detach())
            .map_err(|err| CoreError::Io(format!("failed to write staged tree: {err}")))
    }

    fn tree_id_for_worktree(&self) -> Result<gix::hash::ObjectId, CoreError> {
        let (mut pipeline, index) = self
            .inner
            .filter_pipeline(None)
            .map_err(|err| CoreError::Io(format!("failed to configure filter pipeline: {err}")))?;
        let index_state: &gix::index::State = &index;
        let index_tree_id = self.tree_id_for_index()?;

        let mut editor = self
            .inner
            .edit_tree(index_tree_id)
            .map_err(|err| CoreError::Io(format!("failed to initialize worktree editor: {err}")))?;

        let mut iter = self
            .inner
            .status(gix::progress::Discard)
            .map_err(|err| CoreError::Io(format!("failed to initialize status: {err}")))?
            .index(index.clone())
            .index_worktree_rewrites(None)
            .untracked_files(gix::status::UntrackedFiles::Files)
            .into_index_worktree_iter(Vec::<BString>::new())
            .map_err(|err| CoreError::Io(format!("failed to inspect worktree status: {err}")))?;

        for item in &mut iter {
            let item = item
                .map_err(|err| CoreError::Io(format!("failed to read worktree status: {err}")))?;
            apply_index_worktree_change(&mut pipeline, index_state, &mut editor, item)?;
        }

        if let Some(mut outcome) = iter.into_outcome() {
            if let Some(result) = outcome.write_changes() {
                result.map_err(|err| {
                    CoreError::Io(format!("failed to update index metadata: {err}"))
                })?;
            }
        }

        editor
            .write()
            .map(|id| id.detach())
            .map_err(|err| CoreError::Io(format!("failed to write worktree tree: {err}")))
    }
}

fn signing_tool_missing(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    lower.contains("cannot run gpg")
        || (lower.contains("gpg failed to sign the data")
            && lower.contains("no such file or directory"))
}

fn signing_secret_key_missing(stderr: &str) -> bool {
    let lower = stderr.to_ascii_lowercase();
    lower.contains("no secret key")
        || (lower.contains("inv_sgnr")
            && lower.contains("failure sign")
            && lower.contains("signing failed"))
}

fn append_tree_index_patch(
    repo: &gix::Repository,
    out: &mut String,
    change: gix::diff::index::ChangeRef<'_, '_>,
) -> Result<(), CoreError> {
    match change {
        gix::diff::index::ChangeRef::Addition { location, id, .. } => append_patch(
            repo,
            out,
            None,
            Some(location.as_ref()),
            None,
            Some(id.into_owned()),
        ),
        gix::diff::index::ChangeRef::Deletion { location, id, .. } => append_patch(
            repo,
            out,
            Some(location.as_ref()),
            None,
            Some(id.into_owned()),
            None,
        ),
        gix::diff::index::ChangeRef::Modification {
            location,
            previous_id,
            id,
            ..
        } => append_patch(
            repo,
            out,
            Some(location.as_ref()),
            Some(location.as_ref()),
            Some(previous_id.into_owned()),
            Some(id.into_owned()),
        ),
        gix::diff::index::ChangeRef::Rewrite {
            source_location,
            source_id,
            location,
            id,
            copy,
            ..
        } => {
            if !copy {
                append_patch(
                    repo,
                    out,
                    Some(source_location.as_ref()),
                    None,
                    Some(source_id.into_owned()),
                    None,
                )?;
            }
            append_patch(
                repo,
                out,
                None,
                Some(location.as_ref()),
                None,
                Some(id.into_owned()),
            )
        }
    }
}

fn apply_tree_index_change(
    editor: &mut gix::object::tree::Editor<'_>,
    change: gix::diff::index::ChangeRef<'_, '_>,
) -> Result<(), CoreError> {
    match change {
        gix::diff::index::ChangeRef::Addition {
            location,
            entry_mode,
            id,
            ..
        }
        | gix::diff::index::ChangeRef::Modification {
            location,
            entry_mode,
            id,
            ..
        } => {
            let kind = entry_mode
                .to_tree_entry_mode()
                .ok_or_else(|| CoreError::Io("invalid index entry mode".to_string()))?
                .kind();
            editor
                .upsert(location.as_ref(), kind, id.into_owned())
                .map_err(|err| CoreError::Io(format!("failed to update tree entry: {err}")))?;
        }
        gix::diff::index::ChangeRef::Deletion { location, .. } => {
            editor
                .remove(location.as_ref())
                .map_err(|err| CoreError::Io(format!("failed to remove tree entry: {err}")))?;
        }
        gix::diff::index::ChangeRef::Rewrite {
            source_location,
            location,
            entry_mode,
            id,
            copy,
            ..
        } => {
            if !copy {
                editor
                    .remove(source_location.as_ref())
                    .map_err(|err| CoreError::Io(format!("failed to remove tree entry: {err}")))?;
            }
            let kind = entry_mode
                .to_tree_entry_mode()
                .ok_or_else(|| CoreError::Io("invalid index entry mode".to_string()))?
                .kind();
            editor
                .upsert(location.as_ref(), kind, id.into_owned())
                .map_err(|err| CoreError::Io(format!("failed to update tree entry: {err}")))?;
        }
    }
    Ok(())
}

fn append_index_worktree_patch(
    repo: &gix::Repository,
    pipeline: &mut gix::filter::Pipeline<'_>,
    index: &gix::index::State,
    out: &mut String,
    item: gix::status::index_worktree::Item,
) -> Result<(), CoreError> {
    match item {
        gix::status::index_worktree::Item::Modification {
            entry,
            rela_path,
            status,
            ..
        } => {
            let path = rela_path.as_bstr();
            match status {
                gix::status::plumbing::index_as_worktree::EntryStatus::Change(
                    gix::status::plumbing::index_as_worktree::Change::Removed,
                ) => append_patch(repo, out, Some(path), None, Some(entry.id), None)?,
                gix::status::plumbing::index_as_worktree::EntryStatus::Conflict { .. } => {
                    return Err(CoreError::Io(format!(
                        "cannot diff conflicted index entry at {}",
                        path_display(path)
                    )));
                }
                gix::status::plumbing::index_as_worktree::EntryStatus::NeedsUpdate(_) => {}
                gix::status::plumbing::index_as_worktree::EntryStatus::IntentToAdd
                | gix::status::plumbing::index_as_worktree::EntryStatus::Change(
                    gix::status::plumbing::index_as_worktree::Change::Type { .. }
                    | gix::status::plumbing::index_as_worktree::Change::Modification { .. }
                    | gix::status::plumbing::index_as_worktree::Change::SubmoduleModification(_),
                ) => {
                    if let Some((new_id, _, _)) = pipeline
                        .worktree_file_to_object(path, index)
                        .map_err(map_filter_error)?
                    {
                        append_patch(
                            repo,
                            out,
                            Some(path),
                            Some(path),
                            Some(entry.id),
                            Some(new_id),
                        )?;
                    }
                }
            }
        }
        gix::status::index_worktree::Item::DirectoryContents { entry, .. } => {
            if !matches!(entry.status, gix::dir::entry::Status::Untracked) {
                return Ok(());
            }
            let path = entry.rela_path.as_bstr();
            if let Some((new_id, _, _)) = pipeline
                .worktree_file_to_object(path, index)
                .map_err(map_filter_error)?
            {
                append_patch(repo, out, None, Some(path), None, Some(new_id))?;
            }
        }
        gix::status::index_worktree::Item::Rewrite {
            source,
            dirwalk_entry,
            copy,
            ..
        } => {
            let dst_path = dirwalk_entry.rela_path.as_bstr();
            if !copy {
                if let gix::status::index_worktree::RewriteSource::RewriteFromIndex {
                    source_rela_path,
                    source_entry,
                    ..
                } = source
                {
                    append_patch(
                        repo,
                        out,
                        Some(source_rela_path.as_ref()),
                        None,
                        Some(source_entry.id),
                        None,
                    )?;
                }
            }

            if let Some((new_id, _, _)) = pipeline
                .worktree_file_to_object(dst_path, index)
                .map_err(map_filter_error)?
            {
                append_patch(repo, out, None, Some(dst_path), None, Some(new_id))?;
            }
        }
    }

    Ok(())
}

fn apply_index_worktree_change(
    pipeline: &mut gix::filter::Pipeline<'_>,
    index: &gix::index::State,
    editor: &mut gix::object::tree::Editor<'_>,
    item: gix::status::index_worktree::Item,
) -> Result<(), CoreError> {
    match item {
        gix::status::index_worktree::Item::Modification {
            rela_path, status, ..
        } => {
            let path = rela_path.as_bstr();
            match status {
                gix::status::plumbing::index_as_worktree::EntryStatus::Change(
                    gix::status::plumbing::index_as_worktree::Change::Removed,
                ) => {
                    editor.remove(path).map_err(|err| {
                        CoreError::Io(format!("failed to remove tree entry: {err}"))
                    })?;
                }
                gix::status::plumbing::index_as_worktree::EntryStatus::Conflict { .. } => {
                    return Err(CoreError::Io(format!(
                        "cannot commit with conflicted index entry at {}",
                        path_display(path)
                    )));
                }
                gix::status::plumbing::index_as_worktree::EntryStatus::NeedsUpdate(_) => {}
                gix::status::plumbing::index_as_worktree::EntryStatus::IntentToAdd
                | gix::status::plumbing::index_as_worktree::EntryStatus::Change(
                    gix::status::plumbing::index_as_worktree::Change::Type { .. }
                    | gix::status::plumbing::index_as_worktree::Change::Modification { .. }
                    | gix::status::plumbing::index_as_worktree::Change::SubmoduleModification(_),
                ) => {
                    upsert_worktree_path(pipeline, index, editor, path)?;
                }
            }
        }
        gix::status::index_worktree::Item::DirectoryContents { entry, .. } => {
            if matches!(entry.status, gix::dir::entry::Status::Untracked) {
                upsert_worktree_path(pipeline, index, editor, entry.rela_path.as_bstr())?;
            }
        }
        gix::status::index_worktree::Item::Rewrite {
            source,
            dirwalk_entry,
            copy,
            ..
        } => {
            if !copy {
                if let gix::status::index_worktree::RewriteSource::RewriteFromIndex {
                    source_rela_path,
                    ..
                } = source
                {
                    editor.remove(source_rela_path.as_bstr()).map_err(|err| {
                        CoreError::Io(format!("failed to remove tree entry: {err}"))
                    })?;
                }
            }
            upsert_worktree_path(pipeline, index, editor, dirwalk_entry.rela_path.as_bstr())?;
        }
    }
    Ok(())
}

fn upsert_worktree_path(
    pipeline: &mut gix::filter::Pipeline<'_>,
    index: &gix::index::State,
    editor: &mut gix::object::tree::Editor<'_>,
    path: &BStr,
) -> Result<(), CoreError> {
    let Some((id, kind, _)) = pipeline
        .worktree_file_to_object(path, index)
        .map_err(map_filter_error)?
    else {
        editor
            .remove(path)
            .map_err(|err| CoreError::Io(format!("failed to remove tree entry: {err}")))?;
        return Ok(());
    };

    editor
        .upsert(path, kind, id)
        .map_err(|err| CoreError::Io(format!("failed to update tree entry: {err}")))?;
    Ok(())
}

fn append_patch(
    repo: &gix::Repository,
    out: &mut String,
    old_path: Option<&BStr>,
    new_path: Option<&BStr>,
    old_id: Option<gix::hash::ObjectId>,
    new_id: Option<gix::hash::ObjectId>,
) -> Result<(), CoreError> {
    let old_bytes = read_blob_bytes(repo, old_id)?;
    let new_bytes = read_blob_bytes(repo, new_id)?;

    let old_file = old_path.or(new_path).unwrap_or_else(|| "".into());
    let new_file = new_path.or(old_path).unwrap_or_else(|| "".into());
    let old_disp = path_display(old_file);
    let new_disp = path_display(new_file);

    out.push_str("diff --git ");
    out.push_str(&format!("a/{old_disp} b/{new_disp}\n"));

    match old_path {
        Some(path) => out.push_str(&format!("--- a/{}\n", path_display(path))),
        None => out.push_str("--- /dev/null\n"),
    }
    match new_path {
        Some(path) => out.push_str(&format!("+++ b/{}\n", path_display(path))),
        None => out.push_str("+++ /dev/null\n"),
    }

    let hunks = unified_hunks(&old_bytes, &new_bytes)?;
    out.push_str(&hunks);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(())
}

fn read_blob_bytes(
    repo: &gix::Repository,
    id: Option<gix::hash::ObjectId>,
) -> Result<Vec<u8>, CoreError> {
    let Some(id) = id else {
        return Ok(Vec::new());
    };
    let mut blob = repo
        .find_blob(id)
        .map_err(|err| CoreError::Io(format!("failed to read blob {id}: {err}")))?;
    Ok(blob.take_data())
}

fn unified_hunks(old: &[u8], new: &[u8]) -> Result<String, CoreError> {
    let input = gix::diff::blob::intern::InternedInput::new(old, new);
    let delegate = gix::diff::blob::unified_diff::ConsumeBinaryHunk::new(Vec::<u8>::new(), "\n");
    let sink = gix::diff::blob::UnifiedDiff::new(
        &input,
        delegate,
        gix::diff::blob::unified_diff::ContextSize::symmetrical(3),
    );
    let out = gix::diff::blob::diff(gix::diff::blob::Algorithm::Histogram, &input, sink)
        .map_err(|err| CoreError::Io(format!("failed to build unified hunks: {err}")))?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn path_display(path: &BStr) -> String {
    String::from_utf8_lossy(path.as_ref()).into_owned()
}

fn map_filter_error(err: gix::filter::pipeline::worktree_file_to_object::Error) -> CoreError {
    CoreError::Io(format!(
        "failed to translate worktree file to object: {err}"
    ))
}
