"""SQLAlchemy models for the CommunityTopicRadar warehouse."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by all warehouse models."""


class Post(Base):
    """A question scraped from the community."""

    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body_text: Mapped[str | None] = mapped_column(Text)
    author_id: Mapped[int | None] = mapped_column(Integer)
    author_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    answer_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_answered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    accepted_answer_id: Mapped[int | None] = mapped_column(Integer)
    url: Mapped[str | None] = mapped_column(String(1000))
    source_site: Mapped[str] = mapped_column(String(100), nullable=False)

    __table_args__ = (
        Index("ix_posts_created_at", "created_at"),
        Index("ix_posts_is_answered", "is_answered"),
    )


class Tag(Base):
    """A normalized community tag."""

    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)


class PostTag(Base):
    """Many-to-many association between posts and tags."""

    __tablename__ = "post_tags"

    post_id: Mapped[int] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )


class Author(Base):
    """A community member observed as a question author or answerer."""

    __tablename__ = "authors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    reputation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Answer(Base):
    """An answer observed in the scraped creation window.

    Answers are kept even when their parent question falls outside the
    question sample; they still describe real contributor activity.
    """

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question_id: Mapped[int] = mapped_column(Integer, nullable=False)
    author_id: Mapped[int | None] = mapped_column(Integer)
    author_name: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    score: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source_site: Mapped[str] = mapped_column(String(100), nullable=False)

    __table_args__ = (
        Index("ix_answers_created_at", "created_at"),
        Index("ix_answers_question_id", "question_id"),
    )


class EngagementDaily(Base):
    """One row of daily engagement history per source site."""

    __tablename__ = "engagement_daily"

    day: Mapped[date] = mapped_column(Date, primary_key=True)
    source_site: Mapped[str] = mapped_column(String(100), primary_key=True)
    posts_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    answers_created: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    unique_authors: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    answered_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_views: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
