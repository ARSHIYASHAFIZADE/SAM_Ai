import urllib.request as r
import urllib.error
import json
from http.cookiejar import CookieJar

cj = CookieJar()
opener = r.build_opener(r.HTTPCookieProcessor(cj))

email = "newtest123@test.com"
password = "test"

# Register
print("Registering...")
req_reg = r.Request('https://samai-production.up.railway.app/register', 
                data=json.dumps({"email":email,"password":password}).encode(), 
                headers={'Content-Type':'application/json'})
try:
    resp = opener.open(req_reg)
    print('Register Status:', resp.status)
except urllib.error.HTTPError as e:
    print('Register Error:', e.code, e.read().decode())

# Login
print("Logging in...")
req_log = r.Request('https://samai-production.up.railway.app/login', 
                data=json.dumps({"email":email,"password":password}).encode(), 
                headers={'Content-Type':'application/json'})
try:
    resp = opener.open(req_log)
    print('Login Status:', resp.status)
    print('Cookies:', list(cj))
    
    # Check @me
    print("Checking @me...")
    req_me = r.Request('https://samai-production.up.railway.app/@me')
    resp_me = opener.open(req_me)
    print('@me Status:', resp_me.status)
    print(resp_me.read().decode())
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code, e.read().decode())
except Exception as e:
    print('Error:', e)
