from sqlalchemy import create_engine, Column, Integer, Float, String, Date, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from pathlib import Path
import yaml

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

def get_settings():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def init_engine():
    settings = get_settings()
    engine = create_engine(settings["database_url"], connect_args={"check_same_thread": False} if settings["database_url"].startswith("sqlite") else {})
    SessionLocal.configure(bind=engine)
    return engine

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, autoincrement=True)
    pid = Column(String, unique=True, index=True)  # product id from CSV if available
    title = Column(String, index=True)
    category = Column(String, index=True)
    retail_price = Column(Float)
    discounted_price = Column(Float)
    brand = Column(String, index=True)
    rating = Column(Float)
    url = Column(Text)

    price_history = relationship("PricePoint", back_populates="product", cascade="all, delete-orphan")
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")

class PricePoint(Base):
    __tablename__ = "price_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    date = Column(Date, index=True)
    price = Column(Float)

    product = relationship("Product", back_populates="price_history")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), index=True)
    author = Column(String, default="anon")
    text = Column(Text)
    stars = Column(Float, default=0.0)

    product = relationship("Product", back_populates="reviews")
