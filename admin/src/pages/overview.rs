use askama::Template;
use axum::{extract::State, response::Html};
use chrono::Utc;

use crate::{
    attention::{Attention, AttentionInputs, attention},
    db::{self, DEFAULT_MODE, overview::cumulate},
    error::AppError,
    render::render,
    sidecar,
    state::AdminState,
    view::{money, pct, pnl_class, signed_money, signed_pct},
};

pub struct Kpi {
    pub label: &'static str,
    pub value: String,
    pub class: &'static str,
    pub sub: String,
}

pub struct ActivityView {
    pub time: String,
    pub bot: String,
    pub label: String,
    pub side: String,
    pub price: String,
    pub amount: String,
    pub event: String,
    pub class: &'static str,
}

#[derive(Template)]
#[template(path = "pages/overview.html")]
pub struct OverviewPage {
    pub active: &'static str,
    pub kpis: Vec<Kpi>,
    pub equity_json: String,
    pub attention: Vec<Attention>,
    pub activity: Vec<ActivityView>,
}

pub async fn page(State(st): State<AdminState>) -> Result<Html<String>, AppError> {
    let mode = DEFAULT_MODE;
    let (k, eq, act, times, counts) = tokio::try_join!(
        db::overview::kpis(&st.db, mode),
        db::overview::equity_rows(&st.db, mode),
        db::overview::activity(&st.db, mode),
        db::status::activity_times(&st.db),
        db::status::attention_counts(&st.db, mode),
    )?;
    let model_age = sidecar::health(&st.http, &st.sidecar_url)
        .await
        .ok()
        .and_then(|h| h.model_age_secs);

    let kpis = vec![
        Kpi {
            label: "Bankroll",
            value: money(k.bankroll_trading + k.bankroll_copy),
            class: "",
            sub: format!(
                "ML {} · copy {}",
                money(k.bankroll_trading),
                money(k.bankroll_copy)
            ),
        },
        Kpi {
            label: "Open exposure",
            value: money(k.open_exposure),
            class: "",
            sub: format!("{} positions", k.open_count),
        },
        Kpi {
            label: "PnL 24h",
            value: signed_money(k.pnl_24h),
            class: pnl_class(k.pnl_24h),
            sub: format!("{} resolved", k.resolved_24h),
        },
        Kpi {
            label: "PnL 7d",
            value: signed_money(k.pnl_7d),
            class: pnl_class(k.pnl_7d),
            sub: format!("{} resolved", k.resolved_7d),
        },
        Kpi {
            label: "ROI all-time",
            value: signed_pct(k.roi_all_pct),
            class: pnl_class(k.roi_all_pct.unwrap_or(0.0)),
            sub: signed_money(k.pnl_all),
        },
        Kpi {
            label: "Win rate 30d",
            value: pct(k.win_rate_30d),
            class: "",
            sub: format!("n={}", k.resolved_30d),
        },
    ];

    let attention = attention(&AttentionInputs {
        now: Utc::now(),
        trading_last: times.trading_last,
        copy_last: times.copy_last,
        model_age_secs: model_age,
        open_end_dates: counts.open_end_dates,
        copy_skips_6h: counts.copy_skips_6h,
        ml_bets_24h: counts.ml_bets_24h,
        ml_rejections_24h: counts.ml_rejections_24h,
        top_reject_reason: counts.top_reject_reason,
    });

    let activity = act
        .into_iter()
        .map(|a| {
            let (event, class) = match a.kind.as_str() {
                "won" | "lost" => {
                    let p = a.pnl.unwrap_or(0.0);
                    (format!("{} {}", a.kind, signed_money(p)), pnl_class(p))
                }
                "skipped" => (
                    format!("skipped · {}", a.detail.unwrap_or_default()),
                    "warn",
                ),
                _ => ("placed".to_string(), ""),
            };
            ActivityView {
                time: a.ts.format("%m-%d %H:%M").to_string(),
                bot: a.bot,
                label: a.label,
                side: a.side.to_uppercase(),
                price: format!("{:.2}", a.price),
                amount: money(a.amount),
                event,
                class,
            }
        })
        .collect();

    render(&OverviewPage {
        active: "overview",
        kpis,
        equity_json: serde_json::to_string(&cumulate(&eq)).map_err(anyhow::Error::from)?,
        attention,
        activity,
    })
}
