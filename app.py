# -*- coding: utf-8 -*-
# 추천 학습 데이터 기간 — 기온(대구) · R² 곡선/Top3 표
# 입력 엑셀 포맷 자동 인식:
#  A) Wide형: [연도 | 1월 | ... | 12월]
#  B) Long형: [날짜(datetime) | 평균기온(혹은 임의 이름의 수치열)]
#     - 날짜에서 연/월 추출 → (year, month) 집계
#     - 여러 수치열이 있으면 '평균기온' 이름을 우선, 없으면 첫 번째 수치열 사용

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="추천 학습 데이터 기간 — 기온(대구)", layout="wide")

# -------------------------------
# 공통: 월명 → 1..12 매핑
# -------------------------------
MONTH_ALIASES = {
    "jan":1,"january":1,"1":1,"01":1,"1월":1,
    "feb":2,"february":2,"2":2,"02":2,"2월":2,
    "mar":3,"march":3,"3":3,"03":3,"3월":3,
    "apr":4,"april":4,"4":4,"04":4,"4월":4,
    "may":5,"5":5,"05":5,"5월":5,
    "jun":6,"june":6,"6":6,"06":6,"6월":6,
    "jul":7,"july":7,"7":7,"07":7,"7월":7,
    "aug":8,"august":8,"8":8,"08":8,"8월":8,
    "sep":9,"sept":9,"september":9,"9":9,"09":9,"9월":9,
    "oct":10,"october":10,"10":10,"10월":10,
    "nov":11,"november":11,"11":11,"11월":11,
    "dec":12,"december":12,"12":12,"12월":12,
}

def norm_month(col) -> int|None:
    s = str(col).strip().lower()
    s = s.replace(" ", "")
    s = s.replace("month", "")
    s = s.replace("월평균", "").replace("평균", "")
    if re.fullmatch(r"\d+\s*월", str(col)):
        s = str(col).strip().lower().replace(" ", "").replace("월","")
    return MONTH_ALIASES.get(s, None)

# -------------------------------
# 파서: Wide형 시도
# -------------------------------
def try_parse_wide(df_raw: pd.DataFrame):
    # 헤더 위치 0~4행 스캔
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy()
        body.columns = hdr

        # 첫 열명 표준화
        first = str(body.columns[0]).strip().lower()
        body = body.rename(columns={body.columns[0]: "year"})

        # 월 컬럼 찾기
        month_map = {}
        for c in body.columns[1:]:
            m = norm_month(c)
            if m is not None and 1 <= m <= 12:
                month_map[c] = m
        if len(month_map) >= 6:
            use_cols = ["year"] + list(month_map.keys())
            body = body[use_cols].copy()
            body = body.rename(columns=month_map)
            body["year"] = pd.to_numeric(body["year"], errors="coerce")
            body = body.dropna(subset=["year"])
            body["year"] = body["year"].astype(int)
            long = body.melt(id_vars="year", var_name="month", value_name="temp")
            long["month"] = long["month"].astype(int)
            long["temp"]  = pd.to_numeric(long["temp"], errors="coerce")
            long = long.dropna(subset=["temp"]).sort_values(["year","month"], ignore_index=True)
            if long["year"].nunique() >= 3:
                return long
    return None

# -------------------------------
# 파서: Long형 시도
# -------------------------------
def try_parse_long(df_raw: pd.DataFrame):
    # 헤더 0~4행 탐색
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy()
        body.columns = hdr

        # 날짜 열 찾기
        date_col = None
        for c in body.columns:
            name = str(c).strip().lower()
            if name in ["날짜","date","일자","일시","yyyymmdd","dt"]:
                date_col = c
                break
        # 날짜열 못 찾았으면 "datetime형으로 변환 성공하는 첫 열" 시도
        if date_col is None:
            for c in body.columns:
                s = pd.to_datetime(body[c], errors="coerce")
                if s.notna().sum() >= max(10, int(len(s)*0.2)):
                    date_col = c
                    break
        if date_col is None:
            continue

        # 수치 열 후보: '평균기온' 우선, 없으면 첫 번째 수치열
        value_col = None
        for c in body.columns:
            if str(c).strip() in ["평균기온","기온","temp","temperature"]:
                value_col = c
                break
        if value_col is None:
            numeric_cols = [c for c in body.columns if pd.to_numeric(body[c], errors="coerce").notna().sum()>=max(10, int(len(body)*0.2))]
            # 날짜열 제외
            numeric_cols = [c for c in numeric_cols if c != date_col]
            if numeric_cols:
                value_col = numeric_cols[0]
        if value_col is None:
            continue

        # 정규화
        body = body[[date_col, value_col]].copy()
        body.rename(columns={date_col:"date", value_col:"temp"}, inplace=True)
        body["date"] = pd.to_datetime(body["date"], errors="coerce")
        body["temp"] = pd.to_numeric(body["temp"], errors="coerce")
        body = body.dropna(subset=["date","temp"])
        # 연/월 파생
        body["year"]  = body["date"].dt.year
        body["month"] = body["date"].dt.month

        # 월평균(연,월)
        long = (body.groupby(["year","month"], as_index=False)["temp"]
                     .mean().sort_values(["year","month"], ignore_index=True))
        if long["year"].nunique() >= 3:
            return long
    return None

@st.cache_data
def load_excel_any(path_or_buffer):
    xls = pd.ExcelFile(path_or_buffer)
    # 모든 시트 순회: Wide → Long 순으로 시도
    for sh in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sh, header=None)
        parsed = try_parse_wide(raw)
        if parsed is not None:
            return parsed, sh, "wide"
        parsed = try_parse_long(raw)
        if parsed is not None:
            return parsed, sh, "long"
    return None, None, None

# -------------------------------
# 입력: 업로더 + 기본 파일
# -------------------------------
default_path = Path("기온예측.xlsx")
uploaded = st.file_uploader("월별 평균기온 파일 업로드 (.xlsx)", type=["xlsx"])

df, used_sheet, mode = None, None, None
if uploaded:
    df, used_sheet, mode = load_excel_any(uploaded)
elif default_path.exists():
    df, used_sheet, mode = load_excel_any(default_path)

if df is None:
    st.error("엑셀에서 월별 평균기온을 찾지 못했어.\n"
             "지원 포맷 A) [연도 | 1월..12월], B) [날짜 | 평균기온] 중 하나로 맞춰줘.")
    st.stop()

years = sorted(df["year"].unique().tolist())
min_year, max_year = int(min(years)), int(max(years))

# -------------------------------
# 파라미터
# -------------------------------
colA, colB, colC, colD = st.columns([1,1,1,1.6])
with colA:
    target_year = st.number_input("대상연도(검증용, 실제값 존재)", min_value=min_year+1,
                                  max_value=max_year, value=max_year, step=1)
with colB:
    lower_tail = st.slider("이상저온 컷(하위 p%)", 0, 20, 10, 1,
                           help="월 분포의 하위 p% 값 제외(한파 등 이례치 배제)")
with colC:
    metric_for_top = st.selectbox("Top3 정렬지표", ["R²(높을수록)", "MAE(낮을수록)"], index=0)
with colD:
    note = st.text_input("그래프 상단 제목(옵션)",
                         value=f"추천 학습 데이터 기간 — 최근평년(시트:{used_sheet}, 모드:{mode})")

# -------------------------------
# 유틸
# -------------------------------
def monthwise_recent_mean(train_slice: pd.DataFrame, p_tail: int) -> pd.Series:
    preds = {}
    for m in range(1, 13):
        x = train_slice.loc[train_slice["month"] == m, "temp"].dropna().to_numpy()
        if x.size == 0:
            preds[m] = np.nan
            continue
        if p_tail > 0 and x.size >= 10:
            q = np.percentile(x, p_tail)  # 하위 p%
            x = x[x > q]
        preds[m] = float(np.mean(x)) if x.size else np.nan
    return pd.Series(preds)

def r2_and_mae(y_true: np.ndarray, y_pred: np.ndarray):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y, yp = y_true[mask], y_pred[mask]
    if y.size < 2:
        return np.nan, np.nan
    sse = np.sum((y - yp) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (sse / sst) if sst > 0 else np.nan
    mae = np.mean(np.abs(y - yp))
    return float(r2), float(mae)

# -------------------------------
# 성능 테이블
# -------------------------------
target_vec = df.query("year == @target_year").sort_values("month")["temp"].to_numpy()
candidate_starts = [y for y in years if y < target_year]

rows = []
for s in candidate_starts:
    train = df.query("year >= @s and year <= @(@target_year-1)")
    pred_monthly = monthwise_recent_mean(train, lower_tail).sort_index().to_numpy()
    r2, mae = r2_and_mae(target_vec, pred_monthly)
    rows.append({"시작연도": s, "종료연도": target_year-1, "R2": r2, "MAE": mae})

perf = pd.DataFrame(rows).dropna().sort_values("시작연도").reset_index(drop=True)
if perf.empty:
    st.error("성능표가 비었어. 대상연도/데이터 구간을 확인해줘.")
    st.stop()

top3 = (perf.sort_values("R2", ascending=False).head(3)
        if metric_for_top.startswith("R²") else
        perf.sort_values("MAE", ascending=True).head(3))

# -------------------------------
# 화면
# -------------------------------
st.markdown(f"### {note}")
st.caption(f"대상연도={target_year}, 이상저온 컷={lower_tail}% | 예측치: [시작연도~{target_year-1}] 월평균(하위 p% 제외)")

tbl = top3.copy()
tbl.insert(0, "추천순위", range(1, len(tbl)+1))
tbl["기간"] = tbl.apply(lambda r: f"{int(r['시작연도'])}~현재", axis=1)
tbl = tbl[["추천순위","기간","시작연도","종료연도","R2","MAE"]]
tbl["R2"] = tbl["R2"].map(lambda x: f"{x:.4f}")
tbl["MAE"] = tbl["MAE"].map(lambda x: f"{x:.3f}℃")
st.dataframe(tbl, use_container_width=True)

st.markdown(f"##### 학습 시작연도별 성능(종료연도={target_year-1}) — R² 곡선")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=perf["시작연도"], y=perf["R2"],
    mode="lines+markers+text",
    text=[f"{v:.4f}" if pd.notna(v) else "" for v in perf["R2"]],
    textposition="top center",
    name="R² (train fit)"
))

def add_span(fig, start_y, color, name):
    fig.add_vrect(x0=start_y-0.5, x1=perf["시작연도"].max()+0.5,
                  fillcolor=color, opacity=0.12, line_width=0,
                  annotation_text=name, annotation_position="top left")

palette = ["#4CAF50", "#607D8B", "#009688"]
for i, (_, r) in enumerate(top3.sort_values("시작연도").iterrows()):
    add_span(fig, int(r["시작연도"]), palette[i], f"Top{i+1}: {int(r['시작연도'])}~현재")

s3 = int(target_year - 3)
if s3 in perf["시작연도"].values:
    r2_3 = perf.loc[perf["시작연도"] == s3, "R2"].values[0]
    fig.add_trace(go.Scatter(x=[s3], y=[r2_3], mode="markers",
                             marker=dict(size=14, symbol="star"),
                             name=f"최근3년 시작({s3}~현재)"))

fig.update_yaxes(title="R² (높을수록 적합)",
                 range=[max(0.0, float(perf["R2"].min())-0.01),
                        min(1.0, float(perf["R2"].max())+0.01)])
fig.update_xaxes(title="학습 시작연도(시작연도~현재)")
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

with st.expander("내부 검증 메모 (MAE 요약)"):
    tbl2 = perf.copy()
    tbl2["MAE"] = tbl2["MAE"].map(lambda x: f"{x:.3f}℃")
    tbl2["R2"]  = tbl2["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(tbl2, use_container_width=True)

st.success("최근 연도 중심의 '운영상 최근평년(하위 p% 제외)'이 대상연도 월평균을 가장 잘 근사합니다. 특히 3년 창이 우수합니다.")
