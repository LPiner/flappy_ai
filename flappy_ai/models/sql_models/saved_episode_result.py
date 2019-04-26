from sqlalchemy import Column, Integer, String
from flappy_ai import Base, engine


class SavedEpisodeResult(Base):
    __tablename__ = "episode_results"

    id = Column(Integer, primary_key=True)
    # Not promised to be unique
    episode_number = Column(Integer)
    score = Column(Integer)


SavedEpisodeResult.metadata.create_all(engine)