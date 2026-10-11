//! Pure rules for status pills and the "needs attention" list.

use chrono::{DateTime, NaiveDate, Utc};

use crate::view::{age, ago};

pub const TRADING_STALE_SECS: i64 = 30 * 60;
pub const COPY_STALE_SECS: i64 = 10 * 60;
pub const MODEL_STALE_SECS: f64 = 25.0 * 3600.0;
pub const COPY_SKIPS_6H_LIMIT: i64 = 10;

/// `bets.end_date` is free text from Gamma: RFC3339 or a bare date.
pub fn parse_end_date(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .ok()
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|dt| dt.and_utc())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PillState {
    Ok,
    Stale,
    Down,
}

impl PillState {
    pub fn css(&self) -> &'static str {
        match self {
            PillState::Ok => "g",
            PillState::Stale => "y",
            PillState::Down => "r",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Pill {
    pub name: &'static str,
    pub state: PillState,
    pub detail: String,
}

pub fn bot_pill(
    name: &'static str,
    last: Option<DateTime<Utc>>,
    stale_after_secs: i64,
    now: DateTime<Utc>,
) -> Pill {
    match last {
        None => Pill {
            name,
            state: PillState::Down,
            detail: "no activity".into(),
        },
        Some(t) => {
            let secs = (now - t).num_seconds();
            let state = if secs > stale_after_secs {
                PillState::Stale
            } else {
                PillState::Ok
            };
            Pill {
                name,
                state,
                detail: format!("active {}", ago(secs as f64)),
            }
        }
    }
}

pub fn sidecar_pill(model_age_secs: Option<f64>) -> Pill {
    match model_age_secs {
        None => Pill {
            name: "sidecar",
            state: PillState::Down,
            detail: "unreachable".into(),
        },
        Some(a) if a > MODEL_STALE_SECS => Pill {
            name: "sidecar",
            state: PillState::Stale,
            detail: format!("model {} old", age(a)),
        },
        Some(a) => Pill {
            name: "sidecar",
            state: PillState::Ok,
            detail: format!("model {} old", age(a)),
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    Warn,
    Danger,
}

impl Level {
    pub fn css(&self) -> &'static str {
        match self {
            Level::Warn => "y",
            Level::Danger => "r",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Attention {
    pub level: Level,
    pub text: String,
}

pub struct AttentionInputs {
    pub now: DateTime<Utc>,
    pub trading_last: Option<DateTime<Utc>>,
    pub copy_last: Option<DateTime<Utc>>,
    pub model_age_secs: Option<f64>,
    pub open_end_dates: Vec<Option<String>>,
    pub copy_skips_6h: i64,
    pub ml_bets_24h: i64,
    pub ml_rejections_24h: i64,
    pub top_reject_reason: Option<String>,
}

pub fn attention(i: &AttentionInputs) -> Vec<Attention> {
    let mut out = Vec::new();
    let quiet = |name: &str, last: Option<DateTime<Utc>>, limit: i64| -> Option<Attention> {
        let secs = last.map(|t| (i.now - t).num_seconds());
        match secs {
            Some(s) if s <= limit => None,
            Some(s) => Some(Attention {
                level: Level::Danger,
                text: format!("{name} quiet — last activity {}", ago(s as f64)),
            }),
            None => Some(Attention {
                level: Level::Danger,
                text: format!("{name} quiet — no activity recorded"),
            }),
        }
    };
    out.extend(quiet("trading-bot", i.trading_last, TRADING_STALE_SECS));
    out.extend(quiet("copy-bot", i.copy_last, COPY_STALE_SECS));

    match i.model_age_secs {
        Some(a) if a > MODEL_STALE_SECS => out.push(Attention {
            level: Level::Warn,
            text: format!("Model stale — {} since last retrain", age(a)),
        }),
        None => out.push(Attention {
            level: Level::Danger,
            text: "Model sidecar unreachable".into(),
        }),
        _ => {}
    }

    let horizon = i.now + chrono::Duration::hours(24);
    let expiring = i
        .open_end_dates
        .iter()
        .flatten()
        .filter_map(|s| parse_end_date(s))
        .filter(|d| *d > i.now && *d <= horizon)
        .count();
    if expiring > 0 {
        let noun = if expiring == 1 {
            "position expires"
        } else {
            "positions expire"
        };
        out.push(Attention {
            level: Level::Warn,
            text: format!("{expiring} {noun} in <24h"),
        });
    }

    if i.copy_skips_6h > COPY_SKIPS_6H_LIMIT {
        out.push(Attention {
            level: Level::Danger,
            text: format!("{} copy trades skipped in the last 6h", i.copy_skips_6h),
        });
    }

    if i.ml_bets_24h == 0 && i.ml_rejections_24h > 0 {
        let reason = i.top_reject_reason.as_deref().unwrap_or("unknown");
        out.push(Attention {
            level: Level::Warn,
            text: format!(
                "ML: 0 bets in 24h, {} rejections (top reason: {reason})",
                i.ml_rejections_24h
            ),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 10, 12, 0, 0).unwrap()
    }

    #[test]
    fn parses_rfc3339_date_only_and_rejects_garbage() {
        assert_eq!(
            parse_end_date("2026-10-31T12:00:00Z"),
            Some(Utc.with_ymd_and_hms(2026, 10, 31, 12, 0, 0).unwrap())
        );
        assert_eq!(
            parse_end_date("2026-10-31"),
            Some(Utc.with_ymd_and_hms(2026, 10, 31, 0, 0, 0).unwrap())
        );
        assert_eq!(parse_end_date(""), None);
        assert_eq!(parse_end_date("soon"), None);
    }

    #[test]
    fn bot_pill_states() {
        let fresh = bot_pill(
            "trading-bot",
            Some(now() - chrono::Duration::minutes(3)),
            TRADING_STALE_SECS,
            now(),
        );
        assert_eq!(fresh.state, PillState::Ok);
        assert!(fresh.detail.contains("3m ago"));
        let stale = bot_pill(
            "trading-bot",
            Some(now() - chrono::Duration::hours(2)),
            TRADING_STALE_SECS,
            now(),
        );
        assert_eq!(stale.state, PillState::Stale);
        let none = bot_pill("copy-bot", None, COPY_STALE_SECS, now());
        assert_eq!(none.state, PillState::Down);
        assert_eq!(none.detail, "no activity");
    }

    #[test]
    fn sidecar_pill_states() {
        assert_eq!(sidecar_pill(Some(3.0 * 3600.0)).state, PillState::Ok);
        assert_eq!(sidecar_pill(Some(30.0 * 3600.0)).state, PillState::Stale);
        assert_eq!(sidecar_pill(None).state, PillState::Down);
    }

    fn base_inputs() -> AttentionInputs {
        AttentionInputs {
            now: now(),
            trading_last: Some(now()),
            copy_last: Some(now()),
            model_age_secs: Some(3600.0),
            open_end_dates: vec![],
            copy_skips_6h: 0,
            ml_bets_24h: 1,
            ml_rejections_24h: 0,
            top_reject_reason: None,
        }
    }

    #[test]
    fn healthy_inputs_produce_no_attention() {
        assert!(attention(&base_inputs()).is_empty());
    }

    #[test]
    fn each_rule_fires() {
        let mut i = base_inputs();
        i.trading_last = Some(now() - chrono::Duration::hours(1));
        i.copy_last = None;
        i.model_age_secs = Some(26.0 * 3600.0);
        i.open_end_dates = vec![
            Some("2026-10-10T20:00:00Z".into()), // expires in 8h → counted
            Some("2026-12-01".into()),           // far → not counted
            Some("garbage".into()),              // unparseable → not counted
            None,
        ];
        i.copy_skips_6h = 14;
        i.ml_bets_24h = 0;
        i.ml_rejections_24h = 120;
        i.top_reject_reason = Some("edge_below_floor".into());
        let texts: Vec<String> = attention(&i).into_iter().map(|a| a.text).collect();
        assert!(
            texts.iter().any(|t| t.contains("trading-bot quiet")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("copy-bot quiet")),
            "{texts:?}"
        );
        assert!(texts.iter().any(|t| t.contains("Model stale")), "{texts:?}");
        assert!(
            texts.iter().any(|t| t.contains("1 position expires")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("14 copy trades skipped")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("edge_below_floor")),
            "{texts:?}"
        );
    }
}
