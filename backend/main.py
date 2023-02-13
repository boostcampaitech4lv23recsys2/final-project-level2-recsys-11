from fastapi import FastAPI
import uvicorn

## json float precision 4
import json
class RoundingFloat(float):
    __repr__ = staticmethod(lambda x: format(x, '.4f'))

json.encoder.c_make_encoder = None
if hasattr(json.encoder, 'FLOAT_REPR'):
    # Python 2
    json.encoder.FLOAT_REPR = RoundingFloat.__repr__
else:
    # Python 3
    json.encoder.float = RoundingFloat


try: 
    import pyximport
    pyximport.install()
except:
    pass

from routers import frontend, web4rec, login 

app = FastAPI()

app.include_router(frontend.router)
app.include_router(web4rec.router)
app.include_router(login.router)



if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=30001)