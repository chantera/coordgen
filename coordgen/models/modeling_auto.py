import importlib
from types import ModuleType
from typing import Any, Dict

from transformers import AutoConfig, AutoTokenizer, PretrainedConfig


class _LazyAutoMapping(Dict[str, Any]):
    def __init__(self, mapping: Dict[str, str]):
        super().__init__()
        self._mapping = mapping
        self._modules: Dict[str, ModuleType] = {}

    def __missing__(self, key: str) -> Any:
        if key not in self._mapping:
            raise KeyError(key)

        module_name, attr = self._mapping[key].rsplit(".", 1)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(module_name)

        value = getattr(self._modules[module_name], attr)
        return self.setdefault(key, value)


class _BaseAutoModelClass:
    _model_mapping: _LazyAutoMapping

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated using the "
            f"`{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def _get_model_class(cls, config: PretrainedConfig):
        try:
            return cls._model_mapping[type(config).__name__]
        except KeyError:
            raise ValueError(
                "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
                "Model type should be one of {}.".format(
                    config.__class__,
                    cls.__name__,
                    ", ".join(k for k in cls._model_mapping._mapping.keys()),
                )
            ) from None


MODEL_MAPPING = _LazyAutoMapping(
    {
        "BertConfig": "transformers.models.bert.BertForMaskedLM",
        "T5Config": "coordgen.models.modeling_t5.T5ForConditionalGeneration",
    },
)

MODEL_FOR_COORDINATION_GENERATION_MAPPING = _LazyAutoMapping(
    {
        "BertConfig": "coordgen.models.modeling_bert.BertForCoordinationGeneration",
        "T5Config": "coordgen.models.modeling_t5.T5ForCoordinationGeneration",
    },
)


class _AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                return_unused_kwargs=True,
                **kwargs,
            )
        return cls._get_model_class(config).from_pretrained(
            pretrained_model_name_or_path, *model_args, config=config, **kwargs
        )


class AutoModelForCoordinationGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_COORDINATION_GENERATION_MAPPING
    _auto_model_class = _AutoModel

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model = cls._auto_model_class.from_pretrained(pretrained_model_name_or_path)
        model_class = cls._get_model_class(model.config)
        return model_class(model, tokenizer, **kwargs)
