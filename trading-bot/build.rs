use std::process::Command;

fn main() {
    // Re-run when git state or the generated changelog file changes
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/tags");
    println!("cargo:rerun-if-changed=CHANGELOG.gen");

    // Try reading pre-generated changelog (from CI), fall back to git
    let (tag, changelog) = if let Ok(contents) = std::fs::read_to_string("CHANGELOG.gen") {
        let mut lines = contents.lines();
        let tag = lines.next().unwrap_or("").to_string();
        let log: Vec<&str> = lines.filter(|l| !l.is_empty()).collect();
        (tag, log.join("\n"))
    } else {
        changelog_from_git()
    };

    println!("cargo:rustc-env=BUILD_CHANGELOG={changelog}");
    println!("cargo:rustc-env=BUILD_TAG={tag}");
}

fn changelog_from_git() -> (String, String) {
    let tag = git(&["describe", "--tags", "--abbrev=0"]).unwrap_or_default();

    let range = if tag.is_empty() {
        "HEAD~10..HEAD".to_string()
    } else {
        format!("{tag}..HEAD")
    };

    let log = git(&["log", &range, "--oneline", "--no-decorate"]).unwrap_or_default();

    let bullets: Vec<String> = log
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let msg = l.split_once(' ').map(|(_, m)| m).unwrap_or(l);
            format!("• {msg}")
        })
        .collect();

    let changelog = if bullets.is_empty() {
        "No changes since last release".to_string()
    } else {
        bullets.join("\n")
    };

    (tag, changelog)
}

fn git(args: &[&str]) -> Option<String> {
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}
