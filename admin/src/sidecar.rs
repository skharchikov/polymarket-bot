use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Health {
    pub status: String,
    pub model_loaded: bool,
    pub model_age_secs: Option<f64>,
}

pub async fn health(http: &reqwest::Client, base: &str) -> anyhow::Result<Health> {
    Ok(http
        .get(format!("{}/health", base.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

pub async fn get_json(
    http: &reqwest::Client,
    base: &str,
    path: &str,
) -> anyhow::Result<serde_json::Value> {
    Ok(http
        .get(format!("{}{path}", base.trim_end_matches('/')))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?)
}

/// Flatten a JSON object into sorted `a.b` -> display-string pairs.
/// The retrain-status schema isn't pinned, so the page renders it generically.
pub fn flatten(v: &serde_json::Value) -> Vec<(String, String)> {
    fn walk(prefix: &str, v: &serde_json::Value, out: &mut Vec<(String, String)>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, v) in m {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    walk(&key, v, out);
                }
            }
            serde_json::Value::String(s) => out.push((prefix.to_string(), s.clone())),
            other => out.push((prefix.to_string(), other.to_string())),
        }
    }
    let mut out = Vec::new();
    walk("", v, &mut out);
    out.sort();
    out
}

#[cfg(test)]
mod tests {
    use super::flatten;
    use serde_json::json;

    #[test]
    fn flattens_nested_json_to_dotted_keys() {
        let v = json!({"status": "idle", "last": {"ok": true, "secs": 12.5}, "list": [1, 2]});
        let f = flatten(&v);
        assert!(f.contains(&("status".into(), "idle".into())));
        assert!(f.contains(&("last.ok".into(), "true".into())));
        assert!(f.contains(&("last.secs".into(), "12.5".into())));
        assert!(f.contains(&("list".into(), "[1,2]".into())));
    }
}
