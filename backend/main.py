from router import router
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

def init_app() -> FastAPI:
    # init fast api app
    try:
        # init FastAPI object
        app = FastAPI()

        # middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # router
        app.include_router(router=router)
        return app
    except Exception as e:
        raise ValueError(e.__cause__)

app = init_app()

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        reload_delay=0.5,
        use_colors=True,
    )