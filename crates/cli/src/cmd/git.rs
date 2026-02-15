use std::process::Stdio;

use autocommit_core::CoreError;

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
        self.run_git(&["diff", "--cached"])
    }

    pub(crate) fn diff_worktree(&self) -> Result<String, CoreError> {
        self.run_git(&["diff"])
    }

    pub(crate) fn add_all(&self) -> Result<(), CoreError> {
        self.run_git(&["add", "-A"])?;
        Ok(())
    }

    pub(crate) fn commit(
        &self,
        subject: &str,
        body: Option<&str>,
        no_verify: bool,
    ) -> Result<(), CoreError> {
        let mut owned_args = vec!["commit".to_string(), "-m".to_string(), subject.to_string()];
        if let Some(body) = body {
            owned_args.push("-m".to_string());
            owned_args.push(body.to_string());
        }
        if no_verify {
            owned_args.push("--no-verify".to_string());
        }

        let args = owned_args.iter().map(String::as_str).collect::<Vec<_>>();
        self.run_git(&args)?;
        Ok(())
    }

    pub(crate) fn push(&self) -> Result<(), CoreError> {
        self.run_git(&["push"])?;
        Ok(())
    }

    fn run_git(&self, args: &[&str]) -> Result<String, CoreError> {
        let invocation = format_git_call(args);
        let child = gix::command::prepare("git")
            .with_context(command_context(&self.inner))
            .args(args.iter().copied())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|err| CoreError::Io(format!("failed to run {invocation}: {err}")))?;

        let output = child.wait_with_output().map_err(|err| {
            CoreError::Io(format!("failed to read output for {invocation}: {err}"))
        })?;

        if !output.status.success() {
            return Err(CoreError::Io(format!(
                "{invocation} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}

fn command_context(repo: &gix::Repository) -> gix::command::Context {
    gix::command::Context {
        git_dir: Some(repo.git_dir().to_path_buf()),
        worktree_dir: repo.workdir().map(ToOwned::to_owned),
        no_replace_objects: None,
        ref_namespace: None,
        literal_pathspecs: None,
        glob_pathspecs: None,
        icase_pathspecs: None,
        stderr: None,
    }
}

fn format_git_call(args: &[&str]) -> String {
    format!("git {}", args.join(" "))
}
