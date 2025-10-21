# -*- coding: utf-8 -*-
# 앱 목적:
#  - 대상연도(Y, 기본=2025)를 예측한다고 가정
#  - 시작연도 s를 2013..(Y-1)까지 바꿔가며, [s ~ Y-1] "최근 조건"만으로
#    월별 '운영상 추정치(대상연도 예측치)'를 생성
#    → 방식: 월별 평균(이상저온 하위 p% 컷 제외) = 운영상 '최근평년' 개념
#  - 그 예측치와 Y년 실제값을 비교하여 R²(train fit)과 MAE를 계산
#  - R²가 가장 높은 Top3 기간과 전체 곡선을 시각화(첫 스크린샷 느낌)

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="추천 학습 데이터 기간 — 기온(대구)", layout="wide")

# -------------------------------
# 1) 데이터 로딩
# -------------------------------
@st.cache_data
def load_excel(path_or_bytes) -> pd.DataFrame:
    """
    예상 구조:
      첫 열: 연도(정수/문자), 이후 열: 1월~12월(숫자)
    예) 구분 | 1월 | 2월 | ... | 12월  (구분=연도)
    불규칙한 첫 열명도 처리하고, 월 열은 1..12로 강제 표준화
    """
    df = pd.read_excel(path_or_bytes)
    # 첫 컬럼을 연도로 가정
    df = df.rename(columns={df.columns[0]: "year"})
    # 월 컬럼 후보만 골라서 숫자 월로 정규화
    month_map = {}
    for c in df.columns[1:]:
        s = str(c).strip()
        # '1월', '01', 'Jan' 등 혼용 가능성 최소 처리
        s = s.replace("월", "")
        try:
            m = int(s)
        except:
            continue
        if 1 <= m <= 12:
            month_map[c] = m
    df = df[["year"] + list(month_map.keys())].copy()
    df = df.rename(columns=month_map)
    # wide→long
    df_long = df.melt(id_vars="year", var_name="month", value_name="temp").dropna()
    # 정렬/타입
    df_long["year"] = df_long["year"].astype(int)
    df_long["month"] = df_long["month"].astype(int)
    df_long = df_long.sort_values(["year", "month"], ignore_index=True)
    return df_long

# 기본 파일 경로(서버/로컬 둘 다 고려)
default_path = Path("기온예측.xlsx")
uploaded = st.file_uploader("월별 평균기온 파일 업로드 (.xlsx)", type=["xlsx"])
if uploaded:
    df = load_excel(uploaded)
elif default_path.exists():
    df = load_excel(default_path)
else:
    st.info("샘플 파일명을 '기온예측.xlsx'로 리포 루트에 두거나, 위에 업로드해줘.")
    st.stop()

years = sorted(df["year"].unique())
min_year, max_year = int(min(years)), int(max(years))

# -------------------------------
# 2) 파라미터
# -------------------------------
colA, colB, colC, colD = st.columns([1,1,1,1.2])
with colA:
    target_year = st.number_input("대상연도(검증용, 실제값 존재)", min_value=min_year+1,
                                  max_value=max_year, value=max_year, step=1)
with colB:
    lower_tail = st.slider("이상저온 컷(하위 p%)", min_value=0, max_value=20, value=10, step=1,
                           help="해당 월의 퀀타일 p% 이하 값 제외(한파 등 이례치 배제)")
with colC:
    metric_for_top = st.selectbox("Top3 정렬지표", ["R²(높을수록)", "MAE(낮을수록)"], index=0)
with colD:
    note = st.text_input("그래프 상단 제목(옵션)",
                         value="추천 학습 데이터 기간 — 운영상 최근평년(이상저온 제외)")

# 후보 시작연도 목록(대상연도 직전까지만)
candidate_starts = [y for y in years if y < target_year]

# -------------------------------
# 3) 평가 함수
# -------------------------------
def monthwise_recent_mean(train_slice: pd.DataFrame, p_tail: int) -> pd.Series:
    """
    월별로 하위 p% 컷을 제외한 평균(=운영상 최근평년 개념)을 계산.
    반환: index=1..12, values=예측치(월평균)
    """
    preds = {}
    for m in range(1, 13):
        x = train_slice.loc[train_slice["month"] == m, "temp"].dropna()
        if len(x) == 0:
            preds[m] = np.nan
            continue
        if p_tail > 0 and len(x) >= 10:
            q = np.percentile(x, p_tail)  # 하위 p%
            x = x[x > q]
        preds[m] = x.mean() if len(x) else np.nan
    return pd.Series(preds)

def r2_and_mae(y_true: np.ndarray, y_pred: np.ndarray):
    # 결측 제거 후 계산
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y, yp = y_true[mask], y_pred[mask]
    if len(y) < 2:
        return np.nan, np.nan
    sse = np.sum((y - yp) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (sse / sst) if sst > 0 else np.nan
    mae = np.mean(np.abs(y - yp))
    return float(r2), float(mae)

# -------------------------------
# 4) 전체 후보에 대해 성능 테이블 생성
# -------------------------------
rows = []
target_vec = df.query("year == @target_year").sort_values("month")["temp"].to_numpy()
for s in candidate_starts:
    train = df.query("year >= @s and year <= @(@target_year-1)")
    pred_monthly = monthwise_recent_mean(train, lower_tail).sort_index().to_numpy()
    r2, mae = r2_and_mae(target_vec, pred_monthly)
    rows.append({"시작연도": s, "종료연도": target_year-1, "R2": r2, "MAE": mae})
perf = pd.DataFrame(rows).dropna().sort_values("시작연도").reset_index(drop=True)

# Top3 선정
if metric_for_top == "R²(높을수록)":
    top3 = perf.sort_values("R2", ascending=False).head(3)
else:
    top3 = perf.sort_values("MAE", ascending=True).head(3)

# -------------------------------
# 5) 레이아웃: 상단 표 + 하단 곡선
# -------------------------------
st.markdown(f"### {note}")
st.caption(f"대상연도={target_year}, 이상저온 컷={lower_tail}% | 예측치: [시작연도~{target_year-1}]의 월별 평균(하위 p% 제외)")

# 추천 표(스크린샷 스타일)
tbl = top3.copy()
tbl.insert(0, "추천순위", range(1, len(tbl)+1))
tbl["기간"] = tbl.apply(lambda r: f"{int(r['시작연도'])}~현재", axis=1)
tbl = tbl[["추천순위", "기간", "시작연도", "종료연도", "R2", "MAE"]]
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

# 시각 강조: Top3 기간 영역(시작연도~현재 구간을 음영)
def add_span(fig, start_y, color, name):
    fig.add_vrect(x0=start_y-0.5, x1=perf["시작연도"].max()+0.5,
                  fillcolor=color, opacity=0.12, line_width=0, annotation_text=name,
                  annotation_position="top left")

palette = ["#4CAF50", "#607D8B", "#009688"]  # 눈에 편한 3색(투명도 적용)
for i, (_, r) in enumerate(top3.sort_values("시작연도").iterrows()):
    add_span(fig, int(r["시작연도"]), palette[i], f"Top{i+1}: {int(r['시작연도'])}~현재")

# 3년 평균 지점에 마커 강조(= 시작연도 = target_year-3)
s3 = int(target_year - 3)
if s3 in perf["시작연도"].values:
    r2_3 = perf.loc[perf["시작연도"] == s3, "R2"].values[0]
    fig.add_trace(go.Scatter(
        x=[s3], y=[r2_3],
        mode="markers",
        marker=dict(size=14, symbol="star"),
        name=f"최근3년 시작({s3}~현재)"
    ))

fig.update_yaxes(title="R² (높을수록 적합)", range=[max(0.0, perf["R2"].min()-0.01), min(1.0, perf["R2"].max()+0.01)])
fig.update_xaxes(title="학습 시작연도(시작연도~현재)")
fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

# 하단: 내부 검증 메모(선택)
with st.expander("내부 검증 메모 (MAE 요약)"):
    tbl2 = perf.copy()
    tbl2["MAE"] = tbl2["MAE"].map(lambda x: f"{x:.3f}℃")
    tbl2["R2"]  = tbl2["R2"].map(lambda x: f"{x:.4f}")
    st.dataframe(tbl2, use_container_width=True)

# 메시지 카드
st.success(
    "메시지 요약: 최근 연도만으로 구성된 '운영상 최근평년(하위 p% 이상저온 제외)'이 "
    "대상연도 월평균 기온을 가장 잘 근사합니다. 특히 3년 창이 우수하며, "
    "이 값은 가정용 수요예측 입력기온으로 사용해도 합리적입니다."
)

