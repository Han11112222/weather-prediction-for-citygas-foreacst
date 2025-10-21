# -*- coding: utf-8 -*-
# 심플: 목표연도 Y를 예측할 때, 직전 L년(1~10)의 월별 "단순평균"으로 만든 예측이
# 실제 Y년 월별 기온과 얼마나 유사한지(R²) 비교. 가장 높은 L을 자동 음영 강조.
# 입력 엑셀: Wide([연도 | 1..12]) 또는 Long([날짜 | 평균기온]) 자동 인식.

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="최근 L년 평균 vs 실제 — R² 곡선 (심플·확장)", layout="wide")

# ----------------- 파서 (Wide/Long 자동) -----------------
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
def norm_month(col)->int|None:
    s = str(col).strip().lower().replace(" ", "").replace("month","").replace("월평균","").replace("평균","")
    if re.fullmatch(r"\d+\s*월", str(col)):
        s = str(col).strip().lower().replace(" ","").replace("월","")
    return MONTH_ALIASES.get(s)

def try_parse_wide(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy()
        body.columns = hdr
        body = body.rename(columns={body.columns[0]: "year"})
        month_map = {}
        for c in body.columns[1:]:
            m = norm_month(c)
            if m: month_map[c]=m
        if len(month_map) >= 6:
            use = ["year"] + list(month_map.keys())
            body = body[use].rename(columns=month_map)
            body["year"] = pd.to_numeric(body["year"], errors="coerce")
            body = body.dropna(subset=["year"])
            body["year"] = body["year"].astype(int)
            long = body.melt(id_vars="year", var_name="month", value_name="temp")
            long["month"] = long["month"].astype(int)
            long["temp"]  = pd.to_numeric(long["temp"], errors="coerce")
            long = long.dropna(subset=["temp"]).sort_values(["year","month"])
            if long["year"].nunique() >= 6:
                return long
    return None

def try_parse_long(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy(); body.columns = hdr
        # 날짜 열 찾기
        date_col = None
        for c in body.columns:
            if str(c).strip().lower() in ["날짜","date","일자","일시","dt"]:
                date_col = c; break
        if date_col is None:
            for c in body.columns:
                s = pd.to_datetime(body[c], errors="coerce")
                if s.notna().sum() >= max(12, int(len(s)*0.3)):
                    date_col = c; break
        if date_col is None: continue
        # 값 열: '평균기온' 우선, 없으면 첫 수치열
        val_col = None
        for c in body.columns:
            if str(c).strip() in ["평균기온","기온","temp","temperature"]:
                val_col = c; break
        if val_col is None:
            nums = [c for c in body.columns if c!=date_col and pd.to_numeric(body[c], errors="coerce").notna().sum()>=max(12, int(len(body)*0.3))]
            if nums: val_col = nums[0]
        if val_col is None: continue

        df = body[[date_col, val_col]].copy().rename(columns={date_col:"date", val_col:"temp"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df = df.dropna(subset=["date","temp"])
        df["year"] = df["date"].dt.year
        df["month"]= df["date"].dt.month
        long = (df.groupby(["year","month"], as_index=False)["temp"].mean()
                  .sort_values(["year","month"]))
        if long["year"].nunique() >= 6:
            return long
    return None

@st.cache_data
def load_excel_any(path_or_buf):
    xls = pd.ExcelFile(path_or_buf)
    for sh in xls.sheet_names:
        raw = pd.read_excel(xls, sheet_name=sh, header=None)
        parsed = try_parse_wide(raw)
        if parsed is not None: return parsed, sh, "wide"
        parsed = try_parse_long(raw)
        if parsed is not None: return parsed, sh, "long"
    return None, None, None

# ----------------- 로딩 -----------------
default_path = Path("기온예측.xlsx")
uploaded = st.file_uploader("월별 평균기온 파일 업로드 (.xlsx)", type=["xlsx"])

df, used_sheet, mode = None, None, None
if uploaded:
    df, used_sheet, mode = load_excel_any(uploaded)
elif default_path.exists():
    df, used_sheet, mode = load_excel_any(default_path)

if df is None:
    st.error("엑셀에서 월별 평균기온을 찾지 못했어. 형식: A) [연도|1..12] 또는 B) [날짜|평균기온].")
    st.stop()

years = sorted(df["year"].unique())
min_year, max_year = int(min(years)), int(max(years))

# ----------------- UI -----------------
target_year = st.number_input("대상연도(실제값 존재)", min_value=min_year+1, max_value=max_year, value=max_year, step=1)

# 고정 설명 문구(예시 자동 생성)
ex_start = target_year - 3
ex_end   = target_year - 1
st.markdown(
    f"**설명:** 예를 들어 **{target_year}년**을 예측할 때 **직전 3년 평균**을 쓰면 "
    f"= **{ex_start}~{ex_end}** 월평균으로 {target_year}년 월별을 추정한다는 뜻."
)

st.markdown("### 추천 학습 데이터 기간 — R² 곡선")

# ----------------- 계산 -----------------
end_year = int(target_year - 1)
y_true = df.query("year == @target_year").sort_values("month")["temp"].to_numpy()

def r2(y, yp):
    m = ~(np.isnan(y) | np.isnan(yp))
    y, yp = y[m], yp[m]
    if y.size < 2: return np.nan
    sse = np.sum((y-yp)**2)
    sst = np.sum((y - y.mean())**2)
    return float(1 - sse/sst) if sst>0 else np.nan

rows=[]
for L in range(1, 11):  # 1~10년 창
    start = target_year - L
    if start < min_year:  # 데이터 범위 밖이면 스킵
        continue
    train = df.query("year >= @start and year <= @end_year")
    preds = []
    for m in range(1, 13):
        x = train.loc[train["month"]==m, "temp"].to_numpy()
        preds.append(np.mean(x) if x.size>0 else np.nan)
    y_pred = np.array(preds, dtype=float)
    rows.append((L, r2(y_true, y_pred)))

perf = pd.DataFrame(rows, columns=["L(년)", "R2"]).dropna().sort_values("L(년)")
if perf.empty:
    st.error("계산 가능한 L 구간이 없어. 대상연도·데이터 범위를 확인해줘.")
    st.stop()

best_row = perf.loc[perf["R2"].idxmax()]
best_L, best_R2 = int(best_row["L(년)"]), float(best_row["R2"])

# ----------------- 그래프 -----------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=perf["L(년)"], y=perf["R2"],
    mode="lines+markers+text",
    text=[f"{v:.4f}" for v in perf["R2"]],
    textposition="top center",
    name="R²"
))

# 최적 L 음영 강조(vrect)
fig.add_vrect(
    x0=best_L - 0.5, x1=best_L + 0.5,
    fillcolor="#4CAF50", opacity=0.15, line_width=0,
    annotation_text=f"최적 L={best_L}", annotation_position="top left"
)

# L=3 지점 별표
if 3 in perf["L(년)"].values:
    r2_3 = perf.loc[perf["L(년)"]==3, "R2"].values[0]
    fig.add_trace(go.Scatter(
        x=[3], y=[r2_3], mode="markers",
        marker=dict(size=14, symbol="star"),
        name="L=3(최근3년)"
    ))

# 축/레이아웃
ymin = max(0.0, float(perf["R2"].min()) - 0.01)
ymax = min(1.0, float(perf["R2"].max()) + 0.01)
fig.update_yaxes(title="R² (1에 가까울수록 좋음)", range=[ymin, ymax])
fig.update_xaxes(title=f"직전 L년 평균 (L=1~10), 대상연도={target_year} → 학습 구간: [Y-L .. Y-1]")
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# ----------------- 메시지(한 줄 요약) -----------------
st.success(
    f"요약: **{target_year}년**을 예측할 때, **직전 {best_L}년 평균**이 가장 유사도(R²) 높음 "
    f"(R²={best_R2:.4f})."
)
st.caption(f"(시트: {used_sheet}, 모드: {mode})")
