from fastapi import FastAPI
import uvicorn
import pyximport

pyximport.install()

from routers import database, frontend, web4rec


app = FastAPI()

app.include_router(database.router, prefix='/databsase')
app.include_router(frontend.router, prefix='/frontend')
app.include_router(web4rec.router, prefix='/web4rec-lib')


if __name__=='__main__':
    uvicorn.run(app, host='0.0.0.0', port=30004)