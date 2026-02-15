use std::ops::ControlFlow;

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

    pub(crate) fn commit(
        &self,
        message: &str,
        staged_only: bool,
        _no_verify: bool,
    ) -> Result<(), CoreError> {
        let tree_id = if staged_only {
            self.tree_id_for_index()?
        } else {
            self.tree_id_for_worktree()?
        };

        let parents = self
            .inner
            .head()
            .map_err(|err| CoreError::Io(format!("failed to resolve HEAD: {err}")))?
            .id()
            .map(|id| vec![id.detach()])
            .unwrap_or_default();

        self.inner
            .commit("HEAD", message, tree_id, parents)
            .map_err(|err| CoreError::Io(format!("failed to create commit: {err}")))?;

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

    pub(crate) fn push(&self) -> Result<(), CoreError> {
        Err(CoreError::Io(
            "push is not implemented for the pure-gix backend in this build".to_string(),
        ))
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
