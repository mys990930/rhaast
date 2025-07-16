import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta


# 1. 샘플 OHLCV 데이터 생성 (실제 데이터는 이 부분을 대체하세요)
def create_sample_ohlcv_data(n_periods=100):
    # 기본 날짜 설정
    end_date = datetime.now()
    dates = [end_date - timedelta(days=i) for i in range(n_periods, 0, -1)]

    # 시작 가격
    start_price = 100

    # 가격 생성
    np.random.seed(42)
    closes = [start_price]

    # 랜덤 웤 생성
    for _ in range(n_periods - 1):
        change = np.random.normal(0, 2)
        closes.append(closes[-1] + change)

    # 각 날짜의 OHLCV 데이터 생성
    data = []
    for i in range(n_periods):
        close = closes[i]
        # 고가는 종가보다 0-3% 높게
        high = close * (1 + np.random.uniform(0.005, 0.03))
        # 저가는 종가보다 0-3% 낮게
        low = close * (1 - np.random.uniform(0.005, 0.03))
        # 시가는 고가와 저가 사이의 값
        open_price = np.random.uniform(low, high)
        # 거래량
        volume = np.random.randint(10000, 100000)

        data.append([dates[i], open_price, high, low, close, volume])

    # DataFrame 생성
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    return df


# 2. 실제 OHLCV 데이터를 파일에서 불러오는 함수
def load_ohlcv_from_csv(file_path):
    df = pd.read_csv(file_path)

    # 날짜 열을 인덱스로 설정
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    elif 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)

    # 열 이름 표준화
    col_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns}, inplace=True)

    # 필요한 열만 선택
    required_cols = ['Open', 'High', 'Low', 'Close']
    optional_cols = ['Volume']

    # 모든 필수 열이 있는지 확인
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    # 필수 열만 선택하고 거래량은 있으면 포함
    selected_cols = required_cols + [col for col in optional_cols if col in df.columns]

    return df[selected_cols]


# 3. 캔들스틱 차트 그리기 함수
def plot_candlestick(df, title='OHLCV Candlestick Chart', figsize=(12, 8)):
    """
    OHLCV 데이터로 캔들스틱 차트를 그립니다.

    Args:
        df: pandas DataFrame - OHLCV 데이터
        title: str - 차트 제목
        figsize: tuple - 차트 크기
    """
    # 스타일 설정
    mc = mpf.make_marketcolors(
        up='#26a69a',  # 상승 캔들 색상
        down='#ef5350',  # 하락 캔들 색상
        wick={'up': '#26a69a', 'down': '#ef5350'},  # 심지 색상
        volume={'up': '#26a69a', 'down': '#ef5350'},  # 거래량 색상
        edge={'up': '#26a69a', 'down': '#ef5350'}  # 캔들 테두리 색상
    )

    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        figcolor='white',
        facecolor='white',
        gridcolor='#E0E0E0'
    )

    # 캔들스틱 차트 생성
    kwargs = {
        'type': 'candle',
        'title': title,
        'figratio': (figsize[0] / figsize[1], 1),
        'figscale': 1.2,
        'style': s,
    }

    # 거래량 패널 추가 (거래량 데이터가 있는 경우에만)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        kwargs['volume'] = True
        kwargs['panel_ratios'] = (4, 1)  # 가격:거래량 패널 비율

    # 차트 그리기
    mpf.plot(df, **kwargs)
    plt.tight_layout()
    plt.show()

# 6. 기술적 지표 추가 함수
def plot_candlestick_with_indicators(df, ma_periods=[20, 50], figsize=(12, 8), title='OHLCV with MA, RSI'):
    """
    OHLCV 데이터로 캔들스틱 차트와 기술적 지표(RSI, MACD)를 그립니다.

    Args:
        df: pandas DataFrame - OHLCV 데이터
        title: str - 차트 제목
        ma_periods: list - 이동평균 기간 목록
        figsize: tuple - 차트 크기
    """

    # 이동평균 계산
    df_ma = df.copy()
    for period in ma_periods:
        df_ma[f'MA{period}'] = df['Close'].rolling(window=period).mean()

    # 기술적 지표 계산
    df_ind = df.copy()

    # RSI 계산 (14일)
    delta = df_ind['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df_ind['RSI'] = 100 - (100 / (1 + rs))

    # 스타일 설정
    mc = mpf.make_marketcolors(
        up='#26a69a',
        down='#ef5350',
        wick={'up': '#26a69a', 'down': '#ef5350'},
        volume={'up': '#26a69a', 'down': '#ef5350'},
        edge={'up': '#26a69a', 'down': '#ef5350'}
    )

    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        figcolor='white',
        facecolor='white',
        gridcolor='#E0E0E0'
    )
    # 이동평균선 설정
    ma_colors = ['blue', 'red', 'green', 'purple', 'orange']
    ma_addplots = []

    for i, period in enumerate(ma_periods):
        if i < len(ma_colors):
            color = ma_colors[i]
        else:
            color = f'C{i}'

        ma_addplots.append(
            mpf.make_addplot(df_ma[f'MA{period}'], color=color, width=1.5,
                             legend=f'MA{period}')
        )
    ma_addplots.append(mpf.make_addplot(df_ind['RSI'], panel=1, color='purple', ylabel='RSI'))
    ma_addplots.append(mpf.make_addplot([30] * len(df_ind), panel=1, color='red', linestyle='--'))
    ma_addplots.append(mpf.make_addplot([70] * len(df_ind), panel=1, color='red', linestyle='--'))

    # 캔들스틱 차트 생성
    kwargs = {
        'type': 'candle',
        'title': title,
        'figratio': (figsize[0] / figsize[1], 1),
        'figscale': 1.2,
        'style': s,
        'addplot': ma_addplots,
        'panel_ratios': (6, 2, 2),  # 캔들:RSI:MACD 패널 비율
        'tight_layout': True
    }

    # 거래량 표시 (거래량 데이터가 있는 경우에만)
    if 'Volume' in df_ind.columns and df_ind['Volume'].sum() > 0:
        kwargs['volume'] = True
        kwargs['panel_ratios'] = (6, 1, 2, 2)  # 캔들:거래량:RSI:MACD 패널 비율

    # 차트 그리기
    mpf.plot(df_ind, **kwargs)
    plt.tight_layout()
    plt.show()


# 7. 예제 사용 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    print("샘플 OHLCV 데이터 생성 중...")
    sample_df = create_sample_ohlcv_data(n_periods=100)
    print("데이터 미리보기:")
    print(sample_df.head())

    # 기본 캔들스틱 차트
    print("기본 캔들스틱 차트 생성...")
    plot_candlestick(sample_df)

    # 기술적 지표가 추가된 캔들스틱 차트
    print("기술적 지표가 추가된 캔들스틱 차트 생성...")
    plot_candlestick_with_indicators(sample_df)

    print("모든 차트 생성 완료!")

    # 실제 데이터 사용 예제 (주석 해제하여 사용)
    # csv_path = 'your_data.csv'
    # real_df = load_ohlcv_from_csv(csv_path)
    # plot_candlestick(real_df, title='Real OHLCV Data')
    # plot_candlestick_with_ma(real_df, ma_periods=[20, 50, 200])
    # plot_candlestick_with_bollinger(real_df)
    # plot_candlestick_with_indicators(real_df)