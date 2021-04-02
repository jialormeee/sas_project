import numpy as np 
import pandas as pd 
import ta
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#main driving function 
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    
    settings['day'] += 1
    #print(DATE[-1])
    nMarkets = CLOSE.shape[1]
    #print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))

    if settings['model'] =="technicals":
        return technicals(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] =="svm":
        lookback = settings['lookback']
        dimension = settings['dimension']
        gap = settings['gap']
        pos = np.zeros((1, nMarkets), dtype=np.float)
        momentum = (CLOSE[gap:, :] - CLOSE[:-gap, :]) / CLOSE[:-gap, :]

        for market in range(nMarkets):
            try:
                pos[0, market] = predict(momentum[:, market].reshape(-1, 1),
                                        CLOSE[:, market].reshape(-1, 1),
                                        lookback,
                                        gap,
                                        dimension)
            except ValueError:
                pos[0, market] = .0
        print("Positions:", pos)
        if np.nansum(pos) > 0:
            pos = pos / np.nansum(abs(pos))
        return pos, settings
    
    elif settings['model'] =="linreg":
        return linear_regression(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] =="bollinger":
        nMarkets = len(settings['markets'])
        threshold = settings['threshold']
        pos = np.zeros((1, nMarkets), dtype=np.float)

        for market in range(nMarkets):
            sma, upperBand, lowerBand = bollingerBands(CLOSE[:, market])
            currentPrice = CLOSE[-1, market]

            if currentPrice >= upperBand + (upperBand - lowerBand) * threshold:
                pos[0, market] = -1
            elif currentPrice <= lowerBand - (upperBand - lowerBand) * threshold:
                pos[0, market] = 1
        return pos, settings
    
    elif settings['model'] =="fib_rec":
        return fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] =="volume_method":
        nMarkets=CLOSE.shape[1]
        pos = np.zeros(nMarkets)
        OBVs = [OBV(close,vol) for close,vol in zip(CLOSE,VOL)]
        ###
        def bullish_trend(obv):
            return (obv[-1] > obv[-2]) and (obv[-2] > obv[-3])
        def bearish_trend(obv):
            return (obv[-1] < obv[-2]) and (obv[-2] < obv[-3])

        OBV_bull = [True if bullish_trend(obv) else False for obv in OBVs]
        OBV_bear = [True if bearish_trend(obv) else False for obv in OBVs]
        
        for i in range(0, nMarkets):
            # if bullish take long position
            if OBV_bull[i] == True:
                pos[i+1] = 1
            # if bearish take short position
            elif OBV_bear[i] == True:
                pos[i+1] = -1
        weights = pos/np.nansum(abs(pos))
        return (weights, settings)

#on-balance vol indicator
def OBV(closes, volumes):
    return list(ta.volume.OnBalanceVolumeIndicator(pd.Series(closes), pd.Series(volumes)).on_balance_volume())

#fib-retracement 
def fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    periodLonger=10 #%[280:30:500]
    maxminPeriod=30 
    swing_low = np.nanmin(CLOSE[-maxminPeriod,:],axis=0)
    swing_high = np.nanmax(CLOSE[-maxminPeriod,:])
    pos = np.zeros((1, nMarkets), dtype=np.float)
    diff = swing_high - swing_low

    extremeRange = swing_high - swing_low
    hundred = extremeRange - swing_low
    up_level1 = swing_high - 0.236 * diff
    up_level2 = swing_high - 0.382 * diff
    up_level3 = swing_high - 0.618 * diff

    
    hundred_down = extremeRange + swing_low
    down_level1 = swing_low + 0.236 * diff
    down_level2 = swing_low + 0.382 * diff
    down_level3 = swing_low + 0.618 * diff

    for market in range(nMarkets):
        smaLongerPeriod = np.sum(CLOSE[-periodLonger:,market])/periodLonger
        
        currentPrice = CLOSE[-1, market]

        if currentPrice > smaLongerPeriod and currentPrice<hundred:
            pos[0, market] = -1
        elif currentPrice > smaLongerPeriod and currentPrice<up_level3:
            pos[0, market] = -0.6           
        elif currentPrice > smaLongerPeriod and currentPrice<up_level2:
            pos[0, market] = -0.5
        elif currentPrice > smaLongerPeriod and currentPrice<up_level1:
            pos[0, market] = -0.3
        else:
            if currentPrice < smaLongerPeriod and currentPrice>hundred_down:
                pos[0, market] = 1
            elif currentPrice < smaLongerPeriod and currentPrice>down_level3:
                pos[0, market] = 0.6           
            elif currentPrice < smaLongerPeriod and currentPrice>down_level2:
                pos[0, market] = 0.5
            elif currentPrice < smaLongerPeriod and currentPrice>down_level1:
                pos[0, market] = 0.3


    weights = pos/np.nansum(abs(pos))
    return (weights, settings)

#quantiacs sample code also
def linear_regression(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets = len(settings['markets'])
    lookback = settings['lookback']
    dimension = settings['dimension']
    threshold = settings['threshold']
    pos = np.zeros(nMarkets, dtype=np.float)

    poly = PolynomialFeatures(degree=dimension)
    for market in range(nMarkets):
        reg = linear_model.LinearRegression()
        try:
            reg.fit(poly.fit_transform(np.arange(lookback).reshape(-1, 1)), CLOSE[:, market])
            trend = (reg.predict(poly.fit_transform(np.array([[lookback]]))) - CLOSE[-1, market]) / CLOSE[-1, market]

            if abs(trend[0]) < threshold:
                trend[0] = 0

            pos[market] = np.sign(trend)

        # for NaN data set position to 0
        except ValueError:
            pos[market] = .0

    return pos, settings

#quantiacs sample code also
def bollingerBands(a, n=20):
        sma = np.nansum(a[-n:]) / n
        std = np.std(a[-n:], ddof=1)
        return sma, sma + 2 * std, sma - 2 * std

#quantiacs sample code also
def predict(momentum, CLOSE, lookback, gap, dimension):
    X = np.concatenate([momentum[i:i + dimension] for i in range(lookback - gap - dimension)], axis=1).T
    y = np.sign((CLOSE[dimension+gap:] - CLOSE[dimension+gap-1:-1]).T[0])
    y[y==0] = 1

    clf = svm.SVC()
    clf.fit(X, y)

    return clf.predict(momentum[-dimension:].T)

#added technicals yq
def technicals(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    pos=np.zeros(nMarkets)
    CLOSE = np.transpose(CLOSE)
    VOL = np.transpose(VOL)
    # SMA
    '''
    Baseline indicator
    Compare short-term (sma50) and long-term (sma200) price.
    '''
    sma200=np.nansum(CLOSE[:,-200:],axis=1)/200 
    sma50=np.nansum(CLOSE[:,-50:],axis=1)/50
    

    # MACD
    '''
    Trend indicator
    If the MACD lines are above zero for a sustained period of time,
    the stock is likely trending upwards. Conversely,
    if the MACD lines are below zero for a sustained period of time,
    the trend is likely down.
    '''
    def MACD(closes):
        return list(ta.trend.macd(pd.Series(closes)))

    MACDs = [MACD(close) for close in CLOSE]
    MACDs = np.array([np.array(a) for a in MACDs])
    

    # OBV
    '''
    Trend indicator
    A rising price should be accompanied by a rising OBV;
    a falling price should be accompanied by a falling OBV.
    '''
    def OBV(closes, volumes):
        return list(ta.volume.OnBalanceVolumeIndicator(pd.Series(closes), pd.Series(volumes)).on_balance_volume())

    OBVs = [OBV(close,vol) for close,vol in zip(CLOSE,VOL)]
    OBVs = np.array([np.array(a) for a in OBVs])
    

    # RSI
    '''
    Momentum indicator
    One way to interpret the RSI is by viewing the price as overbought
    when the indicator in the histogram is above 70,
    and viewing the price as oversold when the indicator is below 30.
    '''
    
    def RSI(closes):
        return list(ta.momentum.RSIIndicator(pd.Series(closes)).rsi())

    RSIs = [RSI(close) for close in CLOSE]
    RSIs = np.array([np.array(a) for a in RSIs])
    
    
    # BB
    '''
    Volatility indicator, allowing price to move within a range.
    But if the price is closer to the higher band, it is considered overbought.
    If the price is closer to the lower band, it is considered oversold.
    '''
    def BB_high(closes, period):
        return list(ta.volatility.BollingerBands(pd.Series(closes), period).bollinger_hband_indicator())

    def BB_low(closes, period):
        return list(ta.volatility.BollingerBands(pd.Series(closes), period).bollinger_lband_indicator())

    BBHs = [BB_high(close,20) for close in CLOSE]
    BBHs = np.array([np.array(a) for a in BBHs])

    BBLs = [BB_low(close,20) for close in CLOSE]
    BBLs = np.array([np.array(a) for a in BBLs])


    # Trading Strategy
    '''
    Look at the following:
    baseline is sma50>sma200
    identify upward trend using MACD and OBV
    identify oversold market using RSI and BB
    buy if either baseline or upward trend or oversold is satisfied.
    '''
    uptrend = np.logical_or(np.all(MACDs[:,-7:] > 0), OBVs[:,-1] > OBVs[:,-2])
    oversold = np.logical_or(RSIs[:,-1] < 30, BBLs[:,-1]==1)
    longEquity = np.logical_or(sma50 > sma200, oversold, uptrend)

    pos[longEquity] = 1
    pos[~longEquity] = -1
    
    return pos, settings

def mySettings():
    ''' Define your trading system settings here '''
    markets = ['CASH', 'F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 
    'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 
    'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 
    'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ', 
    'F_GS', 'F_LX', 'F_SS', 'F_DL', 'F_ZQ', 'F_VX', 'F_AE', 'F_BG', 'F_BC', 'F_LU', 'F_DM', 'F_AH', 'F_CF', 'F_DZ', 
    'F_FB', 'F_FL', 'F_FM', 'F_FP', 'F_FY', 'F_GX', 'F_HP', 'F_LR', 'F_LQ', 'F_ND', 'F_NY', 'F_PQ', 'F_RR', 'F_RF', 
    'F_RP', 'F_RY', 'F_SH', 'F_SX', 'F_TR', 'F_EB', 'F_VF', 'F_VT', 'F_VW', 'F_GD', 'F_F']
    
    # MODE = "TEST" / "TRAIN"
    MODE = "TEST"


    train_date = {
        'beginInSample': '19900101',
        'endInSample': '20201231',
    }
    ###this date portion is abit weird 
    test_date = {
        'beginInSample': '20190123',
        'endInSample': '20210331',
    }

    dates = train_date if MODE == "TRAIN" else test_date

    settings = {'markets': markets,
                'lookback': 504,
                'budget': 10 ** 6,
                'slippage': 0.05,
                **dates,
                'day': 0,
                'gap': 20,
                'dimension': 5,
                'threshold': 0.2, ##only bollinger and linreg use threshold
                'model': 'fib_rec' ## model: fib_rec, technicals, 
                }
    

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)