use askama::Template;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
};
use chrono::Utc;
use polymarket_common::storage::portfolio::BetSide;

use super::nonempty;
use crate::{
    attention::parse_end_date,
    db::{self, DEFAULT_MODE, positions::Filter},
    error::AppError,
    render::render,
    state::AdminState,
    view::{age, money, pnl_class, short_wallet, signed_money},
};

pub struct Row {
    pub id: i32,
    pub question: String,
    pub url: String,
    pub bot: String,
    pub who: String,
    pub side: String,
    pub entry: String,
    pub now: String,
    pub cost: String,
    pub unreal: String,
    pub unreal_class: &'static str,
    pub expires: String,
    pub expires_class: &'static str,
}

#[derive(Template)]
#[template(path = "pages/positions.html")]
pub struct PositionsPage {
    pub active: &'static str,
    pub rows: Vec<Row>,
    pub summary: String,
    pub unreal_total: String,
    pub unreal_class: &'static str,
    pub sources: Vec<String>,
    pub strategies: Vec<String>,
    pub f_bot: String,
    pub f_source: String,
    pub f_strategy: String,
}

fn side_of(s: &str) -> BetSide {
    if s.eq_ignore_ascii_case("yes") {
        BetSide::Yes
    } else {
        BetSide::No
    }
}

pub async fn page(
    State(st): State<AdminState>,
    Query(f): Query<Filter>,
) -> Result<Html<String>, AppError> {
    let f = Filter {
        bot: nonempty(f.bot),
        source: nonempty(f.source),
        strategy: nonempty(f.strategy),
    };
    let (rows, (sources, strategies)) = tokio::try_join!(
        db::positions::open_positions(&st.db, DEFAULT_MODE, &f),
        db::positions::filter_options(&st.db, DEFAULT_MODE),
    )?;
    let ids: Vec<String> = rows.iter().map(|r| r.market_id.clone()).collect();
    let prices = st.prices.yes_prices(&ids).await;
    let now = Utc::now();

    let mut exposure = 0.0;
    let mut unreal_total = 0.0;
    let mut any_price = false;
    let view: Vec<Row> = rows
        .into_iter()
        .map(|r| {
            exposure += r.cost + r.fee_paid;
            let side = side_of(&r.side);
            let cur = prices.get(&r.market_id).map(|y| side.price_from_yes(*y));
            // Same definition as early_exit in common/src/storage/postgres.rs.
            let unreal = cur.map(|p| r.shares * p - r.cost);
            if let Some(u) = unreal {
                unreal_total += u;
                any_price = true;
            }
            let end = r.end_date.as_deref().and_then(parse_end_date);
            let left = end.map(|e| (e - now).num_seconds() as f64);
            Row {
                id: r.id,
                question: r.question,
                url: r.url,
                bot: r.bot,
                who: r.trader.map(|t| short_wallet(&t)).unwrap_or(r.strategy),
                side: r.side.to_uppercase(),
                entry: format!("{:.2}", r.entry_price),
                now: cur.map(|p| format!("{p:.2}")).unwrap_or_else(|| "—".into()),
                cost: money(r.cost),
                unreal: unreal.map(signed_money).unwrap_or_else(|| "—".into()),
                unreal_class: unreal.map(pnl_class).unwrap_or(""),
                expires: match left {
                    Some(s) if s <= 0.0 => "ended".into(),
                    Some(s) => age(s),
                    None => "—".into(),
                },
                expires_class: if left.is_some_and(|s| s > 0.0 && s < 86_400.0) {
                    "warn"
                } else {
                    ""
                },
            }
        })
        .collect();

    render(&PositionsPage {
        active: "positions",
        summary: format!("{} open · {} exposure", view.len(), money(exposure)),
        unreal_total: if any_price {
            signed_money(unreal_total)
        } else {
            "—".into()
        },
        unreal_class: if any_price {
            pnl_class(unreal_total)
        } else {
            ""
        },
        rows: view,
        sources,
        strategies,
        f_bot: f.bot.unwrap_or_default(),
        f_source: f.source.unwrap_or_default(),
        f_strategy: f.strategy.unwrap_or_default(),
    })
}

#[derive(Template)]
#[template(path = "partials/position_detail.html")]
pub struct DetailPartial {
    pub d: db::positions::Detail,
}

pub async fn detail(
    State(st): State<AdminState>,
    Path(id): Path<i32>,
) -> Result<Response, AppError> {
    match db::positions::detail(&st.db, id).await? {
        Some(d) => Ok(render(&DetailPartial { d })?.into_response()),
        None => Ok((StatusCode::NOT_FOUND, "bet not found").into_response()),
    }
}
