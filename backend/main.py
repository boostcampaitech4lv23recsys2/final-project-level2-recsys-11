from fastapi import FastAPI
import uvicorn

try: 
    import pyximport
    pyximport.install()
except:
    pass


from routers import frontend, web4rec, login

app = FastAPI()

app.include_router(database.router, prefix='/databsase')
app.include_router(frontend.router,)
app.include_router(web4rec.router, prefix='/web4rec-lib')
app.include_router(login.router, prefix='/user')



if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=30004)