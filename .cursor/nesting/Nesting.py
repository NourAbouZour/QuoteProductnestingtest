import json
import socket
from datetime import datetime
from .NestingCredentials import NestingCredentials

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
        print(f"[DEBUG LOG ERROR] {e}")
# #endregion

class Nesting:
   """NestingCenter API."""

   __serviceUrl = "https://api-nesting.nestingcenter.com/nesting/"

   @staticmethod
   async def start_computation(session, nesting_data):
      """Start a new computation."""
      url = Nesting.__serviceUrl + "start"
      # #region agent log
      _debug_log("Nesting.py:13", "start_computation called", {"service_url": Nesting.__serviceUrl, "full_url": url}, "A")
      # #endregion
      # #region agent log
      try:
         hostname = "api-nesting.nestingcenter.com"
         _debug_log("Nesting.py:15", "attempting DNS resolution", {"hostname": hostname}, "A")
         addr_info = socket.getaddrinfo(hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
         _debug_log("Nesting.py:17", "DNS resolution success", {"addr_count": len(addr_info) if addr_info else 0, "first_addr": str(addr_info[0]) if addr_info else None}, "A")
      except Exception as dns_ex:
         _debug_log("Nesting.py:19", "DNS resolution failed", {"hostname": hostname, "error": str(dns_ex), "error_type": type(dns_ex).__name__}, "A")
      # #endregion
      result = await Nesting.__post(session, url, nesting_data, True)
      if result is not None:
         job_id = result.get('JobId')
         if job_id:
            return Nesting.__serviceUrl + job_id
      raise Exception("Failed to start nesting computation: No JobId returned from API")

   @staticmethod
   async def stop_computation(session, computation_url):
      """Stop a computation."""

      url = computation_url + '/stop'
      return await Nesting.__post(session, url, None, False)

   @staticmethod
   async def computation_messages(session, computation_url):
      """Get all computation messages."""

      url = f'{computation_url}/log'
      return await Nesting.__get(session, url)

   @staticmethod
   async def computation_messages_range(session, computation_url, first, last):
      """Get a range of computation messages."""

      url = f'{computation_url}/log/{first}/{last}'
      return await Nesting.__get(session, url)

   @staticmethod
   async def computation_message(session, computation_url, message_id):
      """Get a computation message."""

      url = f'{computation_url}/log/{message_id}'
      return await Nesting.__get(session, url)

   @staticmethod
   async def computation_status(session, computation_url):
      """Get a computation status"""

      return await Nesting.__get(session, computation_url)

   @staticmethod
   async def computation_stop_details(session, computation_url):
      """Get a computation stop details."""

      url = computation_url + '/stopdetails'
      return await Nesting.__get(session, url)

   @staticmethod
   async def computation_result_last(session, computation_url):
      """Get the last computation result."""
        
      url = f'{computation_url}/result'
      return await Nesting.__get(session, url)

   @staticmethod
   async def computation_result(session, computation_url, result_id):
      """Get a specific computation result."""
        
      if result_id:
         url = f'{computation_url}/result/{result_id}'
         return await Nesting.__get(session, url)
      else:
         return None

   @staticmethod
   async def delete_computation(session, computation_url):
      """Delete a computation sesion."""
        
      try:
         async with session.delete(computation_url, headers=Nesting.__headers()) as response:
            if not response.status < 300:
               Nesting.__print_error(await response.text())
      except Exception as ex:
         Nesting.__print_error(ex)

      return None

   @staticmethod
   def __headers():
      token = NestingCredentials.acquire_token()
      if not token:
         raise Exception("Failed to acquire authentication token. Check credentials in NestingCredentials.py")
      return {'Authorization': 'Bearer ' + token}

   @staticmethod
   async def __post(session, url, nesting_data, get_result):
      try:
         headers = Nesting.__headers()
         async with session.post(url, json=nesting_data, headers=headers) as response:
            response_text = await response.text()
            if response.status < 300:
               if get_result:
                  return json.loads(response_text)
                  #return await response.json()
               return True
            else:
               error_msg = f"HTTP {response.status}: {response_text}"
               Nesting.__print_error(error_msg)
               # Try to parse error details from response
               try:
                  error_json = json.loads(response_text)
                  if 'error' in error_json:
                     error_msg = error_json.get('error_description', error_json.get('error', error_msg))
               except:
                  pass
               raise Exception(f"API request failed: {error_msg}")
      except Exception as ex:
         # #region agent log
         _debug_log("Nesting.py:120", "__post exception caught", {"error": str(ex), "error_type": type(ex).__name__, "error_class": str(type(ex)), "url": url}, "B")
         # #endregion
         # If it's already our custom exception, re-raise it
         if "API request failed" in str(ex) or "authentication token" in str(ex):
            raise
         # Otherwise wrap it
         Nesting.__print_error(ex)
         raise Exception(f"Request failed: {str(ex)}")

   @staticmethod
   async def __get(session, url):
      error_msg = None
      try:
         headers = Nesting.__headers()
         async with session.get(url, headers=headers) as response:
            response_text = await response.text()
            if response.status < 300:
               return json.loads(response_text)
               #return await response.json()
            else:
               error_msg = f"HTTP {response.status}: {response_text}"
               Nesting.__print_error(error_msg)
      except Exception as ex:
         error_msg = str(ex)
         Nesting.__print_error(ex)

      return None

   @staticmethod
   def __print_error(error):
      print(error)




