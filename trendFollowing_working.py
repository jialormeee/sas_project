import numpy as np 
import pandas as pd 
import ta
import statistics as st
from sklearn import svm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle
from pmdarima.arima import auto_arima
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers import Dropout
import tensorflow
from keras.models import model_from_json

#main driving function 
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV, exposure, equity, settings):
    
    settings['day'] += 1
    #print(DATE[-1])
    nMarkets = CLOSE.shape[1]
    print("Using data from {} onwards to predict/take position in {}".format(DATE[0],DATE[-1]))

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

    elif settings['model'] =="fib_rec":
        return fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] =="volume_method":
        nMarkets=CLOSE.shape[1]
        pos = np.zeros(nMarkets)
        OBVs = [OBV(close,vol) for close,vol in zip(CLOSE,VOL)]
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

    elif settings['model'] == 'sarima':
        return sarima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] == 'sarima_auto':
        return sarima_auto(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] == 'sarimax':
        indicators = np.concatenate((USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, USA_JBO, USA_LFPR, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV), axis=1)
        return sarimax(DATE, OPEN, HIGH, LOW, CLOSE, VOL, indicators, exposure, equity, settings)

    elif settings['model'] == 'sarima_tech':
        return sarima_tech(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] == 'moment':
        return moment(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] == 'sarima_industry':
        return sarima_industry(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

    elif settings['model'] == 'lstm':
        return lstm(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings)

def moment(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    lookback=150
    pos = np.zeros((1, nMarkets), dtype=np.float)
    for market in range(nMarkets):
        
        sma_mean = np.mean(np.sum(CLOSE[-lookback:,market])/lookback)
        sma_var = st.stdev(CLOSE[-lookback:, market])
        pastPrice= CLOSE[-lookback,market]
        currentPrice = CLOSE[-1, market]       
        
        ###seeing some price weakness
        if currentPrice<pastPrice:
            ### seeing that price is lower than mean, downward trend, take heavy short position
            if currentPrice < sma_mean:
                pos[0, market] = -1
            ### seeing that price is lower than mean-var, downward trend but slightly more risk-averse, take slightly heavy short position
            elif currentPrice < sma_mean - sma_var:
                pos[0, market] = -0.75
        ### seeing some price strength
        elif currentPrice > pastPrice:
            ### seeing that price is higher than mean, upward trend, take heavy long position
            if currentPrice > sma_mean:
                pos[0, market] = 1
            ### seeing that price is higher than mean + var, upward trend but slightly more risk-averse, take slightly heavy long position
            elif currentPrice > sma_mean + sma_var:
                pos[0, market] = 0.75

    weights = pos/np.nansum(abs(pos))
    return (weights, settings)

#on-balance vol indicator
def OBV(closes, volumes):
    return list(ta.volume.OnBalanceVolumeIndicator(pd.Series(closes), pd.Series(volumes)).on_balance_volume())

#fib-retracement 
def fib_retrac(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    nMarkets=CLOSE.shape[1]
    periodLonger=200
    periodShorter =50 
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

def sarima(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)
    sarima_models = settings['sarima']
    
    for i in range(1, nMarkets):
        model = sarima_models[settings['markets'][i]].fit(np.log(CLOSE[-100:, i]))
        fore = model.predict(1)[0]
        if fore > np.log(CLOSE[-1, i]):
            pos[i] = 1
        else:
            pos[i] = -1

    f = open('weights_list_sarima.txt', 'r')
    weights_list = []
    line = f.readline()
    while len(line) != 0:
        weights_list.append(int(line.strip()))
        line = f.readline()

    weights = np.zeros(nMarkets)
    for i in range(1, nMarkets):
        weights[i] = pos[i]*weights_list[i]/sum(weights_list)

    return weights, settings

def sarima_auto(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)
    # models = {el:None for el in markets[1:]}
    
    for i in range(1, nMarkets):
        # if models[markets[i]] == None:
        model = auto_arima(np.log(CLOSE[:, i]), trace=False, suppress_warnings=True, error_action='ignore')
            # models[markets[i]] = model
        model = model.fit(np.log(CLOSE[:, i]), trace=False, suppress_warnings=True, error_action='ignore')
        fore = model.predict(1)[0]
        if fore > np.log(CLOSE[-1, i]):
            pos[i] = 1
        else:
            pos[i] = -1

    return pos, settings

def sarimax(DATE, OPEN, HIGH, LOW, CLOSE, VOL, indicators, exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)
    with open('sarimax_models.pckl', 'rb') as f:
        sarimax_models = pickle.load(f)

    f = open('weights_list_sarimax.txt', 'r')
    weights_list = []
    line = f.readline()
    while len(line) != 0:
        weights_list.append(int(line.strip()))
        line = f.readline()
    
    for i in range(1, nMarkets):
        if weights_list[i] > 0:
            model = sarimax_models[settings['markets'][i]].fit(np.log(CLOSE[-100:, i]), exogenous = indicators[-100:, :])
            fore = model.predict(1, exogenous = indicators[-1:, :])[0]
            if fore > np.log(CLOSE[-1, i]):
                pos[i] = 1
            else:
                pos[i] = -1

    weights = np.zeros(nMarkets)
    for i in range(1, nMarkets):
        weights[i] = pos[i]*weights_list[i]/sum(weights_list)

    return weights, settings

def sarima_tech(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)
    sarima_models = settings['sarima']
    
    for i in range(1, nMarkets):
        model = sarima_models[settings['markets'][i]].fit(np.log(CLOSE[-100:, i]))
        fore = model.predict(1)[0]
        if fore > np.log(CLOSE[-1, i]):
            pos[i] = 1
        else:
            pos[i] = -1

    f = open('weights_list_sarima.txt', 'r')
    weights_list = []
    line = f.readline()
    while len(line) != 0:
        weights_list.append(int(line.strip()))
        line = f.readline()

    weights = np.zeros(nMarkets)
    for i in range(1, nMarkets):
        weights[i] = pos[i]*weights_list[i]/sum(weights_list)

    '''Adjust weights further using technical indicators'''
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
    Deciding LongEquity:
    baseline is sma50>sma200
    identify upward trend using MACD and OBV
    identify oversold market using RSI and BB
    buy if either baseline or upward trend or oversold is satisfied.
    '''
    uptrend = np.logical_or(np.all(MACDs[:,-7:] > 0), OBVs[:,-1] > OBVs[:,-2])
    oversold = np.logical_or(RSIs[:,-1] < 30, BBLs[:,-1]==1)
    longEquity = np.logical_or(sma50 > sma200, oversold, uptrend)

    '''
    Deciding ShortEquity:
    identify downward trend using MACD and OBV
    identify overbought market using RSI and BB
    sell only when both downward trend and overbought
    '''
    downtrend = np.logical_or(np.all(MACDs[:,-7:] < 0), OBVs[:,-1] < OBVs[:,-2])
    overbought = np.logical_or(RSIs[:,-1] > 70, BBHs[:,-1]==1)
    shortEquity = np.logical_and(downtrend, overbought)

    '''
    Identify buy signal and sell signal using RSI
    buy signal when RSI crosses over to above 50
    sell signal when RSI crosses over to below 50
    '''
    buy_signal = np.logical_and(RSIs[:,-2]<50, RSIs[:,-1]>50)
    sell_signal = np.logical_and(RSIs[:,-2]>50, RSIs[:,-1]<50)

    '''LongEquity, buy_signal, sell_signal -0.3481, 4.1786'''
    for i in range(1, nMarkets):
        if longEquity[i]:
            weights[i] += 0.01
        elif buy_signal[i]:
            weights[i] += 0.005
        elif sell_signal[i]:
            weights[i] -= 0.005

    '''LongEquity, ShortEquity, buy_signal, sell_signal -0.3876, 4.0894
    for i in range(1, nMarkets):
        if longEquity[i]:
            weights[i] += 0.01
        elif buy_signal[i]:
            weights[i] += 0.005
        elif shortEquity[i]:
            weights[i] -= 0.01
        elif sell_signal[i]:
            weights[i] -= 0.005
    '''
    
    return weights, settings

def sarima_industry(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):

    industry_dict_inv = {'F_BC': 'energy', 'F_BG': 'energy', 'F_BO': 'energy', 'F_CL': 'energy', 'F_HO': 'energy', 'F_NG': 'energy', 'F_RB': 'energy', 'F_GC': 'metals', 'F_HG': 'metals', 'F_PA': 'metals', 'F_PL': 'metals', 'F_SI': 'metals', 'F_C': 'agriculture', 'F_CC': 'agriculture', 'F_CT': 'agriculture', 'F_FC': 'agriculture', 'F_KC': 'agriculture', 'F_LB': 'agriculture', 'F_LC': 'agriculture', 'F_LN': 'agriculture', 'F_NR': 'agriculture', 'F_O': 'agriculture', 'F_OJ': 'agriculture', 'F_S': 'agriculture', 'F_SB': 'agriculture', 'F_SM': 'agriculture', 'F_W': 'agriculture', 'F_AE': 'indices', 'F_AH': 'indices', 'F_AX': 'indices', 'F_CA': 'indices', 'F_CF': 'bond', 'F_DM': 'indices', 'F_DX': 'indices', 'F_FB': 'indices', 'F_FP': 'indices', 'F_FY': 'indices', 'F_LX': 'indices', 'F_MD': 'indices', 'F_NQ': 'indices', 'F_NY': 'indices', 'F_RU': 'indices', 'F_SX': 'indices', 'F_VX': 'indices', 'F_YM': 'indices', 'F_XX': 'indices', 'F_EB': 'bond', 'F_F': 'bond', 'F_FV': 'bond', 'F_GS': 'bond', 'F_GX': 'bond', 'F_SS': 'bond', 'F_TU': 'bond', 'F_TY': 'bond', 'F_UB': 'bond', 'F_US': 'bond', 'F_ZQ': 'bond', 'F_AD': 'currency', 'F_BP': 'currency', 'F_CD': 'currency', 'F_ED': 'currency', 'F_JY': 'currency', 'F_LR': 'currency', 'F_MP': 'currency', 'F_ND': 'currency', 'F_RR': 'currency', 'F_SF': 'currency', 'F_TR': 'currency', 'F_EC': 'others', 'F_ES': 'others', 'F_DT': 'others', 'F_UZ': 'others', 'F_DL': 'others', 'F_LU': 'others', 'F_DZ': 'others', 'F_FL': 'others', 'F_FM': 'others', 'F_HP': 'others', 'F_LQ': 'others', 'F_PQ': 'others', 'F_RF': 'others', 'F_RP': 'others', 'F_RY': 'others', 'F_SH': 'others', 'F_VF': 'others', 'F_VT': 'others', 'F_VW': 'others', 'F_GD': 'others'}

    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)
    pos_ind= np.zeros(nMarkets)
    sarima_models = settings['sarima']
    sarima_models_ind = settings['sarima_industry']
    
    for i in range(1, nMarkets):
        model = sarima_models[settings['markets'][i]].fit(np.log(CLOSE[-100:, i]))
        fore = model.predict(1)[0]
        if fore > np.log(CLOSE[-1, i]):
            pos[i] = 1
        else:
            pos[i] = -1

    f = open('weights_list_sarima.txt', 'r')
    weights_list = []
    line = f.readline()
    while len(line) != 0:
        weights_list.append(int(line.strip()))
        line = f.readline()

    for i in range(1, nMarkets):
        model = sarima_models_ind[industry_dict_inv[markets[i]]].fit(np.log(CLOSE[-100:, i]))
        fore = model.predict(1)[0]
        if fore > np.log(CLOSE[-1, i]):
            pos_ind[i] = 1
        else:
            pos_ind[i] = -1

    f = open('weights_list_industry.txt', 'r')
    weights_list_ind = []
    line = f.readline()
    while len(line) != 0:
        weights_list_ind.append(int(line.strip()))
        line = f.readline()

    weights = np.zeros(nMarkets)
    for i in range(1, nMarkets):
        weights[i] = pos[i]*weights_list[i] + pos_ind[i]*weights_list_ind[i]

    return weights, settings

def lstm(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    scaler = MinMaxScaler(feature_range=(0, 1))
    nMarkets = CLOSE.shape[1]
    markets = settings['markets']
    pos= np.zeros(nMarkets)

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    f = open('weights_list_lstm.txt', 'r')
    weights_list = []
    line = f.readline()
    while len(line) != 0:
        weights_list.append(int(line.strip()))
        line = f.readline()

    for i in range(1, nMarkets):
        if weights_list[i] > 0:
            dataset_test = CLOSE[:, i:i+1].astype('float32')
            dataset_test = scaler.fit_transform(dataset_test)
            lookback = 1
            X_test, Y_test = create_dataset(dataset_test, lookback)
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            json_file = open('lstm_models/lstm_model_'+markets[i]+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights('lstm_models/lstm_model_'+markets[i]+'.h5')
            pred = model.predict(X_test)
            pred = scaler.inverse_transform(pred)
            pos[i] = 1 if pred[-1,0] > CLOSE[-1, i] else -1

    weights = np.zeros(nMarkets)
    for i in range(1, nMarkets):
        weights[i] = pos[i]*weights_list[i]/sum(weights_list)

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
        # 'beginInSample': '19900101',
        'beginInSample': '20180101',
        'endInSample': '20201231',
    }
    ###this date portion is abit weird 
    test_date = {
        # 'beginInSample': '20190123',
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
                'model': 'sarima_auto' ## model: fib_rec, technicals, moment, sarima, volume_method
                }

    if settings['model'] == 'sarima' or settings['model'] == 'sarima_industry':
        with open('sarima_models.pckl', 'rb') as f:
            sarima_models = pickle.load(f)
        settings['sarima'] = sarima_models

    if settings['model'] == 'sarima_tech':
        with open('sarima_models.pckl', 'rb') as f:
            sarima_models = pickle.load(f)

        settings['sarima'] = sarima_models

    if settings['model'] == 'sarima_industry':
        with open('sarima_models_ind.pckl', 'rb') as f:
            sarima_models_ind = pickle.load(f)
        settings['sarima_industry'] = sarima_models_ind

    return settings


# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)