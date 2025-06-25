from SmartApi.smartConnect import SmartConnect
import pyotp

def create_session(api_key, client_id, client_pwd, totp_secret):
    obj = SmartConnect(api_key=api_key)  # ✅ Uses the correct key passed from outside

    # Generate TOTP
    totp = pyotp.TOTP(totp_secret).now()

    # Login
    data = obj.generateSession(client_id, client_pwd, totp)

    # Debug print
    print("🔍 Full API Login Response:", data)

    if not data or "data" not in data:
        raise Exception("❌ Login failed: " + str(data))

    # ✅ Use jwtToken instead of accessToken
    access_token = data["data"].get("jwtToken", "").replace("Bearer ", "")
    obj.setAccessToken(access_token)

    return obj



