import base64
import json
from .Nesting import Nesting

class NestingConverters:
   """NestingCenter converters API."""

   __serviceUrl = "https://api-converters.nestingcenter.com/dxfconverter/"
   
   @staticmethod
   async def convert_part(session, drawing_data):
      """Convert a drawing into json part."""

      url = NestingConverters.__serviceUrl + "convertPart"
      json_data = {"ContentBase64": base64.b64encode(drawing_data).decode()}
      
      result = await Nesting._Nesting__post(session, url, json_data, True)
      return result
  
   @staticmethod
   async def create_drawing(session, request_data):
      """Convert a nesting reuslt into dxf drawing."""

      url = NestingConverters.__serviceUrl + "generateReport"
      result = await Nesting._Nesting__post(session, url, request_data, True)
      if result is not None:
         drawing_data = result['ContentBase64']
         if drawing_data:
            return base64.b64decode(drawing_data)
      return None
  
   @staticmethod
   def job_id(computation_url):
      """Get job id."""

      return computation_url[computation_url.rfind("/")+1:]


