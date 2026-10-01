"""
Dataset and Synthetic Creator Video Generator Module.

Provides:
1. VideoRecord: Strongly-typed dataclass representing a YouTube video record.
2. SyntheticVideoDataGenerator: A high-fidelity simulator generating realistic
   creator-economy video metrics, metadata, titles, and synthetic thumbnail images.
3. MultimodalVideoDataset: A PyTorch Dataset designed for joint tabular, text,
   and vision feature ingestion into deep learning fusion architectures.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset


@dataclass
class VideoRecord:
    """Canonical schema for a single creator video observation."""
    video_id: str
    channel_id: str
    channel_title: str
    video_title: str
    thumbnail_url: str
    thumbnail_path: str
    published_at: str
    duration_seconds: float
    tags_count: int
    tags: List[str]
    channel_median_views: float
    video_views: float
    view_ratio: float
    is_outlier: int
    log_view_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary."""
        return asdict(self)


class SyntheticVideoDataGenerator:
    """High-fidelity synthetic data generator for creator video performance modeling.
    
    Simulates:
    - Tiered creator channels (Micro, Mid, Macro, Mega) with baseline variance.
    - Historical video chronologies enforcing a minimum rolling history of 10 videos.
    - Real-world title archetypes (Challenge, Tech Review, Curiosity/Exposé, Tutorial, Vlog).
    - Natural outlier distributions (~18-22% breakout hit rate) influenced by
      title formatting, duration, viral keywords, and timing.
    - Local synthetic thumbnail image generation for end-to-end vision modeling.
    """

    VIRAL_KEYWORDS = [
        "SECRET", "INSANE", "UNBELIEVABLE", "TRUTH", "WARNING", "MISTAKE",
        "STOP", "SHOCKING", "NEVER", "OFFICIAL", "REVEALED", "PROOF",
        "BEST", "WORST", "EXPOSED", "$1,000,000", "50 HOURS", "ULTIMATE"
    ]

    NICHE_DATA = {
        "Tech": {
            "templates": [
                "{brand} {model} After 30 Days: The Honest Truth!",
                "Why I Stopped Using The {product} (And What's Next)",
                "The NEW {product} Just Changed Everything.",
                "{product} vs {competitor}: Don't Make This $1,000 Mistake!",
                "10 INSANE Features You Didn't Know Your {device} Had",
                "Building A $10,000 Custom PC Setup From Scratch",
                "Is {tech_concept} Actually The Future Or Just Hype?",
                "Unboxing The World's Most Expensive {device}",
            ],
            "brands": ["Apple", "Samsung", "Sony", "Google", "Nvidia", "Asus"],
            "models": ["M4 Max", "Ultra", "Pro 2026", "Vision", "Flagship", "RTX 5090"],
            "products": ["MacBook", "iPhone", "Galaxy", "GPU", "OLED Monitor", "Noise Canceling Headphones"],
            "competitors": ["Windows", "iPad", "Android", "AMD", "Dell XPS"],
            "devices": ["Smartphone", "Laptop", "Camera", "Smartwatch"],
            "tech_concepts": ["Quantum Computing", "Local LLMs", "Humane AI", "Foldable Displays"],
            "tag_pool": ["technology", "gadgets", "review", "tech review", "unboxing", "apple", "smartphones", "hardware", "setup"],
            "duration_range": (360, 1800),
        },
        "Challenge": {
            "templates": [
                "I Spent 50 Hours In A {extreme_place}!",
                "Last To Leave The {challenge_zone} Wins ${cash_prize}!",
                "Surviving 7 Days In {survival_place} With Only $10",
                "I Ate Nothing But {weird_food} For A Week (SHOCKING RESULTS)",
                "I Gave 100 Strangers A Choice: Take $100 Or Double It?",
                "Racing An Exotic Supercar vs A Commercial Airplane",
                "I Built An Underground Bunker In 24 Hours",
                "Would You Rather Press This Button Or Win $500,000?",
            ],
            "extreme_places": ["Abandoned Prison", "Haunted Hospital", "Desert Island", "Solitary Confinement"],
            "challenge_zones": ["Giant Red Circle", "Submarine", "Frozen Ice Lake", "Wilderness Island"],
            "cash_prize": ["50,000", "100,000", "250,000", "1,000,000"],
            "survival_places": ["The Amazon Rainforest", "Death Valley", "An Arctic Tundra"],
            "weird_food": ["Military MREs", "Raw Foods", "Gas Station Snacks", "Gold-Plated Food"],
            "tag_pool": ["challenge", "mrbeast", "extreme", "surviving", "cash", "prize", "entertainment", "crazy", "stunt"],
            "duration_range": (600, 2400),
        },
        "Finance": {
            "templates": [
                "How I Built A ${income_stream}/Month Passive Income Portfolio",
                "Why The Stock Market Is Preparing For An Unprecedented Move",
                "The 3 Index Funds That Will Make You A Millionaire",
                "Never Buy A House Before Watching This Video!",
                "How To Legally Pay $0 In Taxes Using Real Estate",
                "My Investment Portfolio Update: What I Bought This Month",
                "5 Money Mistakes You Must Avoid In Your 20s",
                "The Death of The Dollar: What You Must Do Right Now",
            ],
            "income_stream": ["10,000", "25,000", "50,000", "100,000"],
            "tag_pool": ["finance", "investing", "stocks", "real estate", "passive income", "money", "wealth", "economics"],
            "duration_range": (480, 1500),
        },
        "Gaming": {
            "templates": [
                "I Played {game_title} For 100 Days In Hardcore Mode",
                "Why Everyone Is Quitting {game_title} in 2026",
                "Finding The Rarest Easter Egg In {game_title} History",
                "Pro Player vs 100 Beginners in {game_title}",
                "Can You Beat {game_title} Without Taking Any Damage?",
                "Top 10 Secret Features In The NEW {game_title} Update",
                "Trolling toxic players in {game_title} until they rage quit",
                "{game_title} Speedrun World Record Attempt #45",
            ],
            "game_titles": ["Minecraft", "GTA 6", "Elden Ring", "Valorant", "Fortnite", "Call of Duty"],
            "tag_pool": ["gaming", "gameplay", "walkthrough", "speedrun", "hardcore", "multiplayer", "highlights", "funny moments"],
            "duration_range": (540, 3600),
        },
        "Lifestyle": {
            "templates": [
                "A Realistic Day In My Life Living In {city}",
                "Why I Decided To Quit My 9-5 Job And Travel Full Time",
                "My Complete Morning Routine For High Energy & Focus",
                "Moving Into My Dream Apartment in {city} (Empty Tour)",
                "I Tried Waking Up At 5 AM Every Day For 30 Days",
                "Resetting My Life: Deep Cleaning, Organization & New Habits",
                "Weekend Vlog: Cozy Sunday, Grocery Haul & Cooking",
                "Everything I Wish I Knew Before Moving To {city}",
            ],
            "cities": ["Tokyo", "New York City", "London", "Dubai", "Los Angeles", "Paris"],
            "tag_pool": ["vlog", "lifestyle", "day in my life", "morning routine", "apartment tour", "travel", "productivity"],
            "duration_range": (300, 1200),
        },
    }

    CREATOR_TIERS = [
        {"name": "Micro", "weight": 0.40, "median_range": (8_000, 45_000), "volatility": 0.35},
        {"name": "Mid", "weight": 0.35, "median_range": (50_000, 350_000), "volatility": 0.30},
        {"name": "Macro", "weight": 0.20, "median_range": (400_000, 1_800_000), "volatility": 0.25},
        {"name": "Mega", "weight": 0.05, "median_range": (2_000_000, 12_000_000), "volatility": 0.22},
    ]

    def __init__(
        self,
        seed: int = 42,
        num_channels: int = 80,
        outlier_ratio_threshold: float = 2.0,
        thumbnails_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the synthetic data generator.
        
        Args:
            seed: Random seed for reproducibility.
            num_channels: Number of distinct YouTube creator channels to simulate.
            outlier_ratio_threshold: Multiplier R = views / median_views defining an outlier hit.
            thumbnails_dir: Optional directory to save synthetic thumbnail images.
        """
        self.seed = seed
        self.num_channels = num_channels
        self.outlier_ratio_threshold = outlier_ratio_threshold
        self.thumbnails_dir = Path(thumbnails_dir) if thumbnails_dir else Path("data/thumbnails")
        
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _generate_channels(self) -> List[Dict[str, Any]]:
        """Synthesize channel profiles with niches, baselines, and historical stats."""
        channels = []
        niches = list(self.NICHE_DATA.keys())
        tier_choices = [tier["name"] for tier in self.CREATOR_TIERS]
        tier_weights = [tier["weight"] for tier in self.CREATOR_TIERS]

        for i in range(1, self.num_channels + 1):
            channel_id = f"UC_{i:04d}_{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}"
            niche = random.choice(niches)
            tier_name = random.choices(tier_choices, weights=tier_weights, k=1)[0]
            tier_info = next(t for t in self.CREATOR_TIERS if t["name"] == tier_name)
            
            # Baseline median views
            low_m, high_m = tier_info["median_range"]
            baseline_median = float(np.round(np.exp(np.random.uniform(np.log(low_m), np.log(high_m))), -2))
            
            channel_title = f"{tier_name} {niche} Creator {i}"
            channels.append({
                "channel_id": channel_id,
                "channel_title": channel_title,
                "niche": niche,
                "tier": tier_name,
                "baseline_median": baseline_median,
                "volatility": tier_info["volatility"],
            })
        return channels

    def _render_title(self, niche: str) -> str:
        """Render a realistic title based on niche templates and vocabulary."""
        niche_spec = self.NICHE_DATA[niche]
        template = random.choice(niche_spec["templates"])
        
        fillers = {}
        for key in ["brands", "models", "products", "competitors", "devices", 
                    "tech_concepts", "extreme_places", "challenge_zones", 
                    "cash_prize", "survival_places", "weird_food", "income_stream",
                    "game_titles", "cities"]:
            if key in niche_spec:
                fillers[key[:-1] if key.endswith("s") else key] = random.choice(niche_spec[key])
                # Handle specific names in template
                fillers[key] = random.choice(niche_spec[key])
                if key == "cash_prize":
                    fillers["cash_prize"] = random.choice(niche_spec["cash_prize"])
                if key == "income_stream":
                    fillers["income_stream"] = random.choice(niche_spec["income_stream"])
                if key == "game_titles":
                    fillers["game_title"] = random.choice(niche_spec["game_titles"])
                if key == "tech_concepts":
                    fillers["tech_concept"] = random.choice(niche_spec["tech_concepts"])
                if key == "extreme_places":
                    fillers["extreme_place"] = random.choice(niche_spec["extreme_places"])
                if key == "challenge_zones":
                    fillers["challenge_zone"] = random.choice(niche_spec["challenge_zones"])
                if key == "survival_places":
                    fillers["survival_place"] = random.choice(niche_spec["survival_places"])
                if key == "weird_food":
                    fillers["weird_food"] = random.choice(niche_spec["weird_food"])
                if key == "cities":
                    fillers["city"] = random.choice(niche_spec["cities"])
        
        try:
            title = template.format(**fillers)
        except KeyError:
            # Fallback simple title if missing placeholder
            title = f"{niche} Breakdown: 5 Things You Must Know!"

        # Occasional styling heuristics
        r = random.random()
        if r < 0.20:
            title = title.upper()
        elif r < 0.40 and not any(title.endswith(punct) for punct in ["!", "?"]):
            title += " (SHOCKING)"
            
        return title

    def _render_thumbnail(self, video_id: str, title: str, niche: str) -> Tuple[str, str]:
        """Generate a lightweight local thumbnail image and return url & file path."""
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        img_filename = f"{video_id}.jpg"
        img_path = self.thumbnails_dir / img_filename
        
        if not img_path.is_file():
            # Generate deterministic RGB color based on niche and video_id hash
            niche_hues = {
                "Tech": (20, 35, 65),
                "Challenge": (180, 40, 40),
                "Finance": (25, 95, 45),
                "Gaming": (90, 20, 110),
                "Lifestyle": (130, 85, 40),
            }
            base_color = niche_hues.get(niche, (40, 40, 40))
            h_offset = hash(video_id) % 30 - 15
            color = tuple(max(0, min(255, c + h_offset)) for c in base_color)
            
            img = Image.new("RGB", (224, 224), color=color)
            draw = ImageDraw.Draw(img)
            
            # Simple geometric elements to provide visual features for ViT/CLIP
            for step in range(0, 224, 28):
                draw.line([(0, step), (224, step)], fill=(color[0] + 15, color[1] + 15, color[2] + 15), width=1)
                
            # Draw bold central box
            draw.rectangle([(20, 60), (204, 164)], outline=(255, 255, 255), width=3)
            # Draw placeholder text indication
            snippet = title[:16].upper()
            draw.text((28, 90), snippet, fill=(255, 255, 255))
            draw.text((28, 120), niche.upper(), fill=(255, 215, 0))
            
            img.save(img_path, format="JPEG", quality=85)
            
        thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"
        return thumbnail_url, str(img_path.resolve())

    def _calculate_video_metrics(
        self,
        title: str,
        niche: str,
        duration: float,
        channel_median: float,
        channel_volatility: float,
        published_dt: datetime,
    ) -> Tuple[float, float, int, float]:
        """Compute realistic view counts, breakout multiplier, and outlier targets.
        
        Outlier hits (views >= 2.0 * channel_median) are modeled using:
        1. Title hooks (viral keywords, uppercase balance, question/exclamation marks).
        2. Video duration sweet spots (e.g. 10-20 min sweet spot for monetization/retention).
        3. Publish timing (lunchtime / weekend prime windows).
        4. Log-normal heavy-tailed recommendation cascades.
        """
        # Feature heuristics
        upper_ratio = sum(1 for c in title if c.isupper()) / max(len(title), 1)
        has_hook_mark = int("!" in title or "?" in title)
        has_viral_kw = int(any(kw in title.upper() for kw in self.VIRAL_KEYWORDS))
        
        # Viral propensity score (latent affinity)
        score = 0.0
        if 0.15 <= upper_ratio <= 0.60:
            score += 0.35  # Optimal capitalization
        elif upper_ratio > 0.85:
            score += 0.20  # All-caps penalty or niche appeal
            
        if has_hook_mark:
            score += 0.30
        if has_viral_kw:
            score += 0.50
            
        # Timing boost: Weekends (Sat/Sun) and peak upload hours (14:00 - 19:00 UTC)
        if published_dt.weekday() in (5, 6):
            score += 0.15
        if 14 <= published_dt.hour <= 19:
            score += 0.15
            
        # Duration retention sweet spot: 8 mins to 25 mins (480s - 1500s)
        if 480 <= duration <= 1500:
            score += 0.25
        elif duration < 60:
            score += 0.10  # Shorts volatility
            
        # Realistic heavy-tailed multiplier
        # Mean log ratio centered near 0 (R=1.0) with latent score shift and volatility
        log_mean = -0.15 + (score * 0.45)
        log_std = channel_volatility + 0.35
        
        # Generate raw multiplier from log-normal distribution
        raw_multiplier = np.random.lognormal(mean=log_mean, sigma=log_std)
        
        # Inject occasional viral breakout cascades (long right-tail hit)
        if np.random.random() < (0.08 + score * 0.08):
            viral_cascade_multiplier = np.random.uniform(2.2, 5.5)
            raw_multiplier = max(raw_multiplier, viral_cascade_multiplier)
            
        # Enforce realistic bounds
        raw_multiplier = float(np.clip(raw_multiplier, 0.08, 12.0))
        
        video_views = float(np.round(channel_median * raw_multiplier))
        view_ratio = float(video_views / max(channel_median, 1.0))
        is_outlier = int(view_ratio >= self.outlier_ratio_threshold)
        log_view_ratio = float(np.log1p(view_ratio))
        
        return video_views, view_ratio, is_outlier, log_view_ratio

    def generate_records(self, total_samples: int = 2500) -> List[VideoRecord]:
        """Generate a full catalog of synthetic creator video records.
        
        Args:
            total_samples: Approximate number of total videos to generate across channels.
            
        Returns:
            List of strongly-typed VideoRecord instances.
        """
        channels = self._generate_channels()
        videos_per_channel = max(15, math.ceil(total_samples / len(channels)))
        
        records: List[VideoRecord] = []
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        counter = 1
        for ch in channels:
            channel_id = ch["channel_id"]
            channel_title = ch["channel_title"]
            niche = ch["niche"]
            baseline_median = ch["baseline_median"]
            volatility = ch["volatility"]
            niche_spec = self.NICHE_DATA[niche]
            
            # Simulate historical chronological uploads
            curr_time = base_time + timedelta(days=random.randint(0, 30))
            channel_history_views: List[float] = []
            
            for v_idx in range(videos_per_channel):
                video_id = f"vid_{counter:06d}"
                counter += 1
                
                # Advance time realistically (every 3 to 10 days)
                curr_time += timedelta(
                    days=random.randint(2, 8),
                    hours=random.randint(1, 23),
                    minutes=random.randint(0, 59)
                )
                published_at_str = curr_time.isoformat()
                
                # Title and duration
                title = self._render_title(niche)
                dur_min, dur_max = niche_spec["duration_range"]
                duration = float(np.random.randint(dur_min, dur_max))
                
                # Tags
                tags_count = random.randint(6, 25)
                niche_tags = random.sample(
                    niche_spec["tag_pool"], 
                    k=min(tags_count, len(niche_spec["tag_pool"]))
                )
                extra_tags = [f"tag_{random.randint(100, 999)}" for _ in range(tags_count - len(niche_tags))]
                tags = niche_tags + extra_tags
                
                # Channel rolling median: uses past video views if at least 10 exist, else baseline
                if len(channel_history_views) >= 10:
                    rolling_median = float(np.median(channel_history_views[-15:]))
                else:
                    rolling_median = baseline_median
                    
                # Compute views and outlier targets
                views, ratio, is_outlier, log_ratio = self._calculate_video_metrics(
                    title=title,
                    niche=niche,
                    duration=duration,
                    channel_median=rolling_median,
                    channel_volatility=volatility,
                    published_dt=curr_time,
                )
                channel_history_views.append(views)
                
                # Render thumbnail mock image
                thumbnail_url, thumbnail_path = self._render_thumbnail(video_id, title, niche)
                
                record = VideoRecord(
                    video_id=video_id,
                    channel_id=channel_id,
                    channel_title=channel_title,
                    video_title=title,
                    thumbnail_url=thumbnail_url,
                    thumbnail_path=thumbnail_path,
                    published_at=published_at_str,
                    duration_seconds=duration,
                    tags_count=tags_count,
                    tags=tags,
                    channel_median_views=rolling_median,
                    video_views=views,
                    view_ratio=ratio,
                    is_outlier=is_outlier,
                    log_view_ratio=log_ratio,
                )
                records.append(record)
                
                if len(records) >= total_samples:
                    break
            if len(records) >= total_samples:
                break
                
        # Shuffle records to avoid strict channel clustering during standard processing
        random.shuffle(records)
        return records

    def generate_dataframe(self, total_samples: int = 2500) -> pd.DataFrame:
        """Generate dataset directly as a pandas DataFrame."""
        records = self.generate_records(total_samples=total_samples)
        return pd.DataFrame([r.to_dict() for r in records])

    def save_to_json(self, output_path: Union[str, Path], total_samples: int = 2500) -> Path:
        """Generate and serialize dataset into a JSON file."""
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        records = self.generate_records(total_samples=total_samples)
        data = [r.to_dict() for r in records]
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return out_file


def generate_synthetic_dataset(
    output_path: Union[str, Path] = "data/sample_dataset.json",
    num_samples: int = 2500,
    num_channels: int = 80,
    seed: int = 42,
    thumbnails_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Convenience helper to generate and save synthetic dataset."""
    generator = SyntheticVideoDataGenerator(
        seed=seed,
        num_channels=num_channels,
        thumbnails_dir=thumbnails_dir,
    )
    generator.save_to_json(output_path=output_path, total_samples=num_samples)
    return load_video_dataset(output_path)


def load_video_dataset(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load JSON video dataset into a typed DataFrame."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    # Ensure types
    df["duration_seconds"] = df["duration_seconds"].astype(float)
    df["tags_count"] = df["tags_count"].astype(int)
    df["channel_median_views"] = df["channel_median_views"].astype(float)
    df["video_views"] = df["video_views"].astype(float)
    df["view_ratio"] = df["view_ratio"].astype(float)
    df["is_outlier"] = df["is_outlier"].astype(int)
    df["log_view_ratio"] = df["log_view_ratio"].astype(float)
    return df


class MultimodalVideoDataset(Dataset):
    """PyTorch Dataset supporting tabular features, text, vision, and dual targets."""

    def __init__(
        self,
        tabular_features: np.ndarray,
        titles: List[str],
        thumbnail_paths: List[str],
        labels_cls: np.ndarray,
        labels_reg: np.ndarray,
        video_ids: Optional[List[str]] = None,
        precomputed_text_embeddings: Optional[np.ndarray] = None,
        precomputed_vision_embeddings: Optional[np.ndarray] = None,
        image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the PyTorch Multimodal Dataset.
        
        Args:
            tabular_features: Normalized tabular matrix of shape (N, D_tab).
            titles: List of video titles of length N.
            thumbnail_paths: List of thumbnail image paths of length N.
            labels_cls: Binary outlier hit indicator of shape (N,).
            labels_reg: Continuous log1p(R) targets of shape (N,).
            video_ids: Optional list of unique video identifiers.
            precomputed_text_embeddings: Optional precomputed array (N, D_text).
            precomputed_vision_embeddings: Optional precomputed array (N, D_vis).
            image_transform: Optional image transform callable for raw PIL images.
        """
        self.tabular_features = torch.tensor(tabular_features, dtype=torch.float32)
        self.titles = list(titles)
        self.thumbnail_paths = list(thumbnail_paths)
        self.labels_cls = torch.tensor(labels_cls, dtype=torch.float32).unsqueeze(1)
        self.labels_reg = torch.tensor(labels_reg, dtype=torch.float32).unsqueeze(1)
        
        self.video_ids = video_ids if video_ids is not None else [f"vid_{i}" for i in range(len(titles))]
        
        self.text_embeddings = (
            torch.tensor(precomputed_text_embeddings, dtype=torch.float32)
            if precomputed_text_embeddings is not None else None
        )
        self.vision_embeddings = (
            torch.tensor(precomputed_vision_embeddings, dtype=torch.float32)
            if precomputed_vision_embeddings is not None else None
        )
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.titles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {
            "video_id": self.video_ids[idx],
            "tabular": self.tabular_features[idx],
            "label_cls": self.labels_cls[idx],
            "label_reg": self.labels_reg[idx],
            "title": self.titles[idx],
            "thumbnail_path": self.thumbnail_paths[idx],
        }

        # Include text representation
        if self.text_embeddings is not None:
            item["text_embedding"] = self.text_embeddings[idx]

        # Include vision representation
        if self.vision_embeddings is not None:
            item["vision_embedding"] = self.vision_embeddings[idx]
        elif self.image_transform is not None:
            img_path = self.thumbnail_paths[idx]
            try:
                with Image.open(img_path) as img:
                    item["vision_image"] = self.image_transform(img.convert("RGB"))
            except Exception:
                # Fallback zero tensor if image missing
                item["vision_image"] = torch.zeros((3, 224, 224), dtype=torch.float32)

        return item
