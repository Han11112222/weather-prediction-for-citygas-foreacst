# -*- coding: utf-8 -*-
# 목적: 최근 몇 년 데이터를 사용할수록 R²가 어떻게 변하는지 "한 장"으로 보여주기
# - 입력 엑셀: Wide([연도|1~12월]) 또는 Long([날짜|평균기온]) 자동 인식
# - 이상저온 하위 p% 컷 제외 평균으로 '운영상 최근평년' 계산
# - 화면: R² 곡선 하나만 (라벨 표시 + 최근3년 지점 별표)

from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="추천 학습 데이터 기간 — R² 곡선", layout="wide")

# --------------- 유틸 ---------------
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
    s = str(col).strip().lower()
    s = s.replace(" ", "").replace("month","").replace("월평균","").replace("평균","")
    if re.fullmatch(r"\d+\s*월", str(col)):
        s = str(col).strip().lower().replace(" ","").replace("월","")
    return MONTH_ALIASES.get(s)

def _r2_mae(y, yp):
    mask = ~(np.isnan(y) | np.isnan(yp))
    y, yp = y[mask], yp[mask]
    if y.size < 2: return np.nan, np.nan
    sse = np.sum((y-yp)**2); sst = np.sum((y - y.mean())**2)
    r2 = 1 - sse/sst if sst>0 else np.nan
    mae = np.mean(np.abs(y-yp))
    return float(r2), float(mae)

def _recent_mean(train, p_tail):
    preds = {}
    for m in range(1,13):
        x = train.loc[train["month"]==m, "temp"].dropna().to_numpy()
        if x.size==0:
            preds[m]=np.nan; continue
        if p_tail>0 and x.size>=10:
            q = np.percentile(x, p_tail)
            x = x[x>q]
        preds[m]=float(np.mean(x)) if x.size else np.nan
    return pd.Series(preds)

# --------------- 파서 (Wide/Long 자동) ---------------
def try_parse_wide(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy()
        body.columns = hdr
        body = body.rename(columns={body.columns[0]: "year"})
        month_map={}
        for c in body.columns[1:]:
            m = norm_month(c)
            if m: month_map[c]=m
        if len(month_map)>=6:
            use = ["year"]+list(month_map.keys())
            body = body[use].rename(columns=month_map)
            body["year"] = pd.to_numeric(body["year"], errors="coerce")
            body = body.dropna(subset=["year"])
            body["year"] = body["year"].astype(int)
            long = body.melt(id_vars="year", var_name="month", value_name="temp")
            long["month"]=long["month"].astype(int)
            long["temp"] = pd.to_numeric(long["temp"], errors="coerce")
            long = long.dropna(subset=["temp"]).sort_values(["year","month"])
            if long["year"].nunique()>=3: return long
    return None

def try_parse_long(df_raw: pd.DataFrame):
    for header_row in range(0, min(5, len(df_raw))):
        hdr = list(df_raw.iloc[header_row])
        body = df_raw.iloc[header_row+1:].copy(); body.columns = hdr
        # 날짜열 찾기
        date_col=None
        for c in body.columns:
            if str(c).strip().lower() in ["날짜","date","일자","일시","dt"]: date_col=c; break
        if date_col is None:
            for c in body.columns:
                s=pd.to_datetime(body[c], errors="coerce")
                if s.notna().sum()>=max(10,int(len(s)*0.2)): date_col=c; break
        if date_col is None: continue
        # 값열: '평균기온' 우선
        val_col=None
        for c in body.columns:
            if str(c).strip() in ["평균기온","기온","temp","temperature"]: val_col=c; break
        if val_col is None:
            nums=[c for c in body.columns if c!=date_col and pd.to_numeric(body[c], errors="coerce").notna().sum()>=max(10,int(len(body)*0.2))]
            if nums: val_col=nums[0]
        if val_col is None: continue
        df = body[[date_col,val_col]].copy().rename(columns={date_col:"date", val_col:"temp"})
        df["date"]=pd.to_datetime(df["date"], errors="coerce"); df["temp"]=pd.to_numeric(df["temp"], errors="coerce")
        df=df.dropna(subset=["date","temp"])
        df["year"]=df["date"].dt.year; df["month"]=df["date"].dt.month
        long=(df.groupby(["year","month"],as_index=False)["temp"].mean()).sort_values(["year","month"])
        if long["year"].nunique()>=3: return long
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

# --------------- 입력 ---------------
default_path = Path("기온예측.xlsx")
uploaded = st.file_uploader("월별 평균기온 파일 업로드 (.xlsx)", type=["xlsx"])

df, used_sheet, mode = None, None, None
if uploaded:
    df, used_sheet, mode = load_excel_any(uploaded)
elif default_path.exists():
    df, used_sheet, mode = load_excel_any(default_path)

if df is None:
    st.error("엑셀에서 월별 평균기온을 찾지 못했어. 형식: A) [연도|1~12월] 또는 B) [날짜|평균기온].")
    st.stop()

years = sorted(df["year"].unique().tolist())
min_year, max_year = int(min(years)), int(max(years))

# 최소 UI (대상연도·컷만)
col1, col2 = st.columns([1,2])
with col1:
    target_year = st.number_input("대상연도", min_value=min_year+1, max_value=max_year, value=max_year, step=1)
with col2:
    p_tail = st.slider("이상저온 컷(하위 p%)", 0, 20, 10)

# --------------- 성능 계산 ---------------
end_year = int(target_year - 1)              # << 핵심 수정: query에 직접 산술식 넣지 말고 변수화
target_vec = df.query("year == @target_year").sort_values("month")["temp"].to_numpy()

rows=[]
for s in [y for y in years if y < target_year]:
    train = df.query("year >= @s and year <= @end_year")  # << 안전한 query 구문
    pred = _recent_mean(train, p_tail).sort_index().to_numpy()
    r2, mae = _r2_mae(target_vec, pred)
    rows.append((s, r2, mae))
perf = pd.DataFrame(rows, columns=["start","R2","MAE"]).dropna().sort_values("start")
if perf.empty:
    st.error("성능표가 비었어. 대상연도/데이터 확인.")
    st.stop()

# --------------- 그래프 (R² 곡선만) ---------------
st.markdown("### 추천 학습 데이터 기간 — R² 곡선")
st.caption(f"(시트: {used_sheet}, 모드: {mode})  ·  대상연도={target_year}, 종료연도={end_year}, 컷={p_tail}%")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=perf["start"], y=perf["R2"],
    mode="lines+markers+text",
    text=[f"{v:.4f}" if pd.notna(v) else "" for v in perf["R2"]],
    textposition="top center",
    name="R²"
))

# 최근 3년 시작점 별표
s3 = int(target_year - 3)
if s3 in perf["start"].values:
    r2_3 = perf.loc[perf["start"]==s3, "R2"].values[0]
    fig.add_trace(go.Scatter(x=[s3], y=[r2_3], mode="markers",
                             marker=dict(size=14, symbol="star"),
                             name=f"최근3년 시작({s3}~{end_year})"))

fig.update_yaxes(title="R² (높을수록 적합)", range=[max(0.0, float(perf["R2"].min())-0.01), min(1.0, float(perf["R2"].max())+0.01)])
fig.update_xaxes(title="학습 시작연도 (시작연도~현재)")
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)
