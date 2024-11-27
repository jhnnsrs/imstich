from mikro_next.rath import MikroNextRath
from mikro_next.traits import HasZarrStoreTrait, HasZarrStoreAccessor
from rath.scalars import ID
from enum import Enum
from mikro_next.scalars import FourByFourMatrix
from mikro_next.funcs import aexecute, execute
from typing import Optional, Tuple, Literal
from pydantic import ConfigDict, BaseModel, Field


class GetStageViewsQueryStageAffineviewsImageStore(HasZarrStoreAccessor, BaseModel):
    typename: Literal["ZarrStore"] = Field(
        alias="__typename", default="ZarrStore", exclude=True
    )
    id: ID
    key: str
    "The key where the data is stored."
    bucket: str
    "The bucket where the data is stored."
    model_config = ConfigDict(frozen=True)


class GetStageViewsQueryStageAffineviewsImage(HasZarrStoreTrait, BaseModel):
    typename: Literal["Image"] = Field(
        alias="__typename", default="Image", exclude=True
    )
    id: ID
    name: str
    "The name of the image"
    store: GetStageViewsQueryStageAffineviewsImageStore
    "The store where the image data is stored."
    model_config = ConfigDict(frozen=True)


class GetStageViewsQueryStageAffineviews(BaseModel):
    typename: Literal["AffineTransformationView"] = Field(
        alias="__typename", default="AffineTransformationView", exclude=True
    )
    id: ID
    affine_matrix: FourByFourMatrix = Field(alias="affineMatrix")
    image: GetStageViewsQueryStageAffineviewsImage
    model_config = ConfigDict(frozen=True)


class GetStageViewsQueryStage(BaseModel):
    typename: Literal["Stage"] = Field(
        alias="__typename", default="Stage", exclude=True
    )
    affine_views: Tuple[GetStageViewsQueryStageAffineviews, ...] = Field(
        alias="affineViews"
    )
    model_config = ConfigDict(frozen=True)


class GetStageViewsQuery(BaseModel):
    stage: GetStageViewsQueryStage

    class Arguments(BaseModel):
        stage_id: ID = Field(alias="stageId")

    class Meta:
        document = "query GetStageViews($stageId: ID!) {\n  stage(id: $stageId) {\n    affineViews {\n      id\n      affineMatrix\n      image {\n        id\n        name\n        store {\n          id\n          key\n          bucket\n          __typename\n        }\n        __typename\n      }\n      __typename\n    }\n    __typename\n  }\n}"


async def aget_stage_views(
    stage_id: ID, rath: Optional[MikroNextRath] = None
) -> GetStageViewsQueryStage:
    """GetStageViews


    Arguments:
        stage_id (ID): The unique identifier of an object
        rath (mikro_next.rath.MikroNextRath, optional): The mikro rath client

    Returns:
        GetStageViewsQueryStage
    """
    return (await aexecute(GetStageViewsQuery, {"stageId": stage_id}, rath=rath)).stage


def get_stage_views(
    stage_id: ID, rath: Optional[MikroNextRath] = None
) -> GetStageViewsQueryStage:
    """GetStageViews


    Arguments:
        stage_id (ID): The unique identifier of an object
        rath (mikro_next.rath.MikroNextRath, optional): The mikro rath client

    Returns:
        GetStageViewsQueryStage
    """
    return execute(GetStageViewsQuery, {"stageId": stage_id}, rath=rath).stage
