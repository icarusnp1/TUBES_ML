from app.preprocessors.standard import StandardPreprocessor
from app.preprocessors.minmax import MinMaxPreprocessor
from app.preprocessors.robust import RobustPreprocessor
from app.cluster_models.agglomerative import AgglomerativeModel

PREPROCESSORS = {
    StandardPreprocessor.id: StandardPreprocessor(),
    MinMaxPreprocessor.id: MinMaxPreprocessor(),
    RobustPreprocessor.id: RobustPreprocessor(),
}

CLUSTER_MODELS = {
    AgglomerativeModel.id: AgglomerativeModel(),
}


DEFAULT_GAMING_FEATURES = [
    "Age",
    "PlayTimeHours",
    "InGamePurchases",
    "GameDifficulty",
    "SessionsPerWeek",
    "AvgSessionDurationMinutes",
    "PlayerLevel",
    "AchievementsUnlocked"
]
