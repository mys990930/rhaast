import ccxt
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = str(os.environ.get('apiKey'))
SECRET_KEY = str(os.environ.get('secret'))

binance_session = ccxt.binance(config={
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
    'options':{
        'defaultType': 'future',
        'adjustForTimeDifference': True,
    }
})