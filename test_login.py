from login_helper import create_session

api_key = "IbvVzG5v"
client_id = "J57353407"
client_pwd = "2222"
totp_secret = "CGPGDARIZ6EE2C555AR2RZDZ2Q"

smartapi = create_session(api_key, client_id, client_pwd, totp_secret)
print("✅ Login successful")
