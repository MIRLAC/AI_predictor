from SmartApi.smartConnect import SmartConnect
import pyotp

def create_session(client_id, client_pwd, totp_secret, market_feed_key, historical_feed_key):
    obj = SmartConnect(api_key=market_feed_key)  # api_key here is misleading, it's the feed key
    
    token = pyotp.TOTP(totp_secret).now()
    data = obj.generateSession(client_id, client_pwd, token)

    if "data" not in data:
        raise Exception("❌ Login Failed", data)

    return obj




