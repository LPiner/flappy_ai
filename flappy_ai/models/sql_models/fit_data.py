from sqlalchemy import Column, Integer, String, Float
from flappy_ai import Base, engine


class FitData(Base):
    __tablename__ = "fit_data"

    id = Column(Integer, primary_key=True)
    epsilon = Column(Float)
    loss = Column(Float)
    accuracy = Column(Float)


FitData.metadata.create_all(engine)
