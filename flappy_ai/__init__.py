from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import os

db_path = Path(f"{os.path.dirname(__file__)}/../data/data.db")


Base = declarative_base()
engine = create_engine(f'sqlite:///{db_path}')

Session = sessionmaker(bind=engine)

