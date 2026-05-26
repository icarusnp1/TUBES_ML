from app.preprocessors.standard import StandardPreprocessor
from app.preprocessors.minmax import MinMaxPreprocessor
from app.preprocessors.robust import RobustPreprocessor
from app.preprocessors.mean_drop_minmax import MeanDropMinMaxPreprocessor
from app.preprocessors.mean_zscore import MeanZScorePreprocessor
from app.cluster_models.agglomerative import AgglomerativeModel
from app.cluster_models.dbscan import DBSCANModel
from app.cluster_models.kmeans import KMeansModel

PREPROCESSORS = {
    StandardPreprocessor.id: StandardPreprocessor(),
    MinMaxPreprocessor.id: MinMaxPreprocessor(),
    RobustPreprocessor.id: RobustPreprocessor(),
    MeanDropMinMaxPreprocessor.id: MeanDropMinMaxPreprocessor(),
    MeanZScorePreprocessor.id: MeanZScorePreprocessor(),
}

CLUSTER_MODELS = {
    AgglomerativeModel.id: AgglomerativeModel(),
    DBSCANModel.id: DBSCANModel(),
    KMeansModel.id: KMeansModel(),
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
