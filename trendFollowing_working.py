import numpy as np 
import pandas as pd 
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import ta

#main driving function 
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    
    settings['day'] += 1
    print(DATE[-1])
    nMarkets = CLOSE.shape[1]

    if settings['strategy'] =="sma":
        return trend_following(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] =="svm":
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
    
    elif settings['strategy'] =="linreg":
        return linear_regression(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] =="bollinger":
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
    
    elif settings['strategy'] =="fib_rec":
        return fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['strategy'] =="volume_method":
        return avg6mthvol(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

def avg6mthvol(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    period = int(365/2)
    #find top vol over period of 1/2 year and return corresponding market
    
    nMarkets = len(settings['markets'])
    pos = np.zeros((1, nMarkets), dtype=np.float)

    latestVol = VOL[-period:, :]
    #find average volume over 6 mth per market
    aveVol = np.mean(latestVol,axis=0) # list of nMaekts with average over prev days 
    
    #find average volume over 6 mth per market s
    aveVol = np.mean(latestVol,axis=0) # list of nMaekts with average over prev days 
    longE = np.where(aveVol == np.nanmax(aveVol))
    shortE = np.where(aveVol == np.nanmin(aveVol))

    pos[0,longE[0]] = 1
    pos[0,shortE[0]] = -1
    
    weights = pos/np.nansum(abs(pos))
    return weights, settings

def fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    
    periodLonger=300 #%[280:30:500]
    maxminPeriod=60 
    price_min = np.nanmin(CLOSE[-maxminPeriod,:],axis=0)
    price_max = np.nanmax(CLOSE[-maxminPeriod,:])
    pos = np.zeros((1, nMarkets), dtype=np.float)
    diff = price_max - price_min

    extremeRange = price_max - price_min
    hundred = extremeRange - price_min
    up_level1 = price_max - 0.236 * diff
    up_level2 = price_max - 0.382 * diff
    up_level3 = price_max - 0.618 * diff

    
    hundred_down = extremeRange + price_min
    down_level1 = price_min + 0.236 * diff
    down_level2 = price_min + 0.382 * diff
    down_level3 = price_min + 0.618 * diff

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

#original model given by quantiacs 
def trend_following(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    periodLonger = 200
    periodShorter = 40
    nMarkets = CLOSE.shape[1]
    # Calculate Simple Moving Average (SMA)
    smaLongerPeriod = np.nansum(CLOSE[-periodLonger:, :], axis=0) / periodLonger
    smaShorterPeriod = np.nansum(CLOSE[-periodShorter:, :], axis=0) / periodShorter

    longEquity = smaShorterPeriod > smaLongerPeriod
    shortEquity = ~longEquity

    pos = np.zeros(nMarkets)
    pos[longEquity] = 1
    pos[shortEquity] = -1

    weights = pos / np.nansum(abs(pos))

    return weights, settings


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
                'strategy': 'volume_method', ## strategy: sma, svm, bollinger, fib_rec, linreg
                }

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)