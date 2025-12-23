# Required packages: msal
from msal import PublicClientApplication
import json
import os
from datetime import datetime

# #region agent log
def _debug_log(location, message, data, hypothesis_id):
    try:
        log_path = r"c:\Users\User\Desktop\QuoteProduct-132d0ac7cb48648e235b5458dc590025a7b7d3c0\.cursor\debug.log"
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
            f.flush()
    except Exception as e:
        # Don't fail silently - at least print to console for debugging
        print(f"[DEBUG LOG ERROR] {e}")
# #endregion

class NestingCredentials:
   """NestingCenter API."""

   __serviceId = "14bc1c96-d677-4a08-8a07-68725b6bd732"
   __authority2 = "https://starsoftonline.b2clogin.com/tfp/starsoftonline.onmicrosoft.com/B2C_1_Sign"
   __authority = "https://starsoftonline.b2clogin.com/tfp/starsoftonline.onmicrosoft.com/B2C_1_ROPC"
   __apiScopes = ["https://starsoftonline.onmicrosoft.com/81588f52-db64-40bd-8096-e75159abdd9a/NestingCenter"]
   __client_application = PublicClientApplication(
        client_id =__serviceId,
        authority =__authority)

   @staticmethod
   def acquire_token2():
       token = NestingCredentials.__acquire_token_silent()
       if token is None:
           token = NestingCredentials.__acquire_token_interactive()
       return token
   
   @staticmethod
   def acquire_token():
      # #region agent log
      _debug_log("NestingCredentials.py:22", "acquire_token called", {}, "A")
      # #endregion
      token = NestingCredentials.__acquire_token_silent()
      # #region agent log
      _debug_log("NestingCredentials.py:24", "silent token result", {"token_is_none": token is None, "token_length": len(token) if token else 0}, "A")
      # #endregion
      if token is None:
          # #region agent log
          _debug_log("NestingCredentials.py:25", "calling ROPC fallback", {}, "B")
          # #endregion
          token = NestingCredentials.__acquire_token_ropc()
          # #region agent log
          _debug_log("NestingCredentials.py:26", "ROPC token result", {"token_is_none": token is None, "token_length": len(token) if token else 0}, "B")
          # #endregion
      # #region agent log
      _debug_log("NestingCredentials.py:27", "acquire_token returning", {"token_is_none": token is None, "token_length": len(token) if token else 0}, "A")
      # #endregion
      return token

   @staticmethod
   def __acquire_token_silent():
      # #region agent log
      _debug_log("NestingCredentials.py:29", "__acquire_token_silent entry", {}, "C")
      # #endregion
      accounts = NestingCredentials.__client_application.get_accounts()
      # #region agent log
      _debug_log("NestingCredentials.py:31", "get_accounts result", {"account_count": len(accounts) if accounts else 0}, "C")
      # #endregion
      if accounts:
         auth_result = NestingCredentials.__client_application.acquire_token_silent(
                          NestingCredentials.__apiScopes, accounts[0])
         # #region agent log
         _debug_log("NestingCredentials.py:33", "silent auth result", {"has_access_token": "access_token" in auth_result if auth_result else False, "has_error": "error" in auth_result if auth_result else False, "error": auth_result.get("error") if auth_result and "error" in auth_result else None}, "C")
         # #endregion
         try:
            return auth_result["access_token"]
         except KeyError:
            # #region agent log
            _debug_log("NestingCredentials.py:36", "KeyError in silent auth", {"auth_result_keys": list(auth_result.keys()) if auth_result else []}, "C")
            # #endregion
            return None
      # #region agent log
      _debug_log("NestingCredentials.py:38", "no accounts, returning None", {}, "C")
      # #endregion
      return None

   @staticmethod
   def __acquire_token_interactive():
      auth_result = NestingCredentials.__client_application.acquire_token_interactive(NestingCredentials.__apiScopes)

      try:
         return auth_result["access_token"]
      except KeyError:
         print("Error: " + auth_result.get("error"))
         print("Error description: " + auth_result.get("error_description"))
      return None
  
   @staticmethod
   def __acquire_token_ropc():
      # #region agent log
      _debug_log("NestingCredentials.py:52", "__acquire_token_ropc entry", {}, "D")
      # #endregion
      user = "nour.abouzour@naggiar.net"
      psw = "nour@1234"
      # #region agent log
      _debug_log("NestingCredentials.py:54", "credentials loaded", {"user": user, "user_is_none": user is None, "psw_length": len(psw) if psw else 0}, "D")
      # #endregion

      if user == None:
         raise ValueError("Please set username and password in the NestingCredentials.py class.")

      # #region agent log
      _debug_log("NestingCredentials.py:59", "calling acquire_token_by_username_password", {"authority": NestingCredentials.__authority, "client_id": NestingCredentials.__serviceId}, "D")
      # #endregion
      auth_result = NestingCredentials.__client_application.acquire_token_by_username_password(user, psw, NestingCredentials.__apiScopes)
      # #region agent log
      _debug_log("NestingCredentials.py:59", "ROPC auth_result received", {"has_access_token": "access_token" in auth_result if auth_result else False, "has_error": "error" in auth_result if auth_result else False, "error": auth_result.get("error") if auth_result and "error" in auth_result else None, "error_description": auth_result.get("error_description") if auth_result and "error_description" in auth_result else None, "all_keys": list(auth_result.keys()) if auth_result else []}, "D")
      # #endregion

      try:
         token = auth_result["access_token"]
         # #region agent log
         _debug_log("NestingCredentials.py:62", "ROPC success, returning token", {"token_length": len(token) if token else 0}, "D")
         # #endregion
         return token
      except KeyError:
         # #region agent log
         _debug_log("NestingCredentials.py:64", "KeyError in ROPC - no access_token", {"auth_result_keys": list(auth_result.keys()) if auth_result else [], "error": auth_result.get("error") if auth_result else None}, "E")
         # #endregion
         print("Error: " + auth_result.get("error"))
         print("Error description: " + auth_result.get("error_description"))
      # #region agent log
      _debug_log("NestingCredentials.py:66", "ROPC returning None", {}, "E")
      # #endregion
      return None



