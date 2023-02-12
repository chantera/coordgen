import importlib

from transformers.models.auto import auto_factory, configuration_auto, tokenization_auto


class _LazyAutoMapping(auto_factory._LazyAutoMapping):
    def _load_attr_from_module(self, model_type, attr):
        if "." in attr:
            module_name, attr = attr.rsplit(".", 1)
        else:
            module_name = configuration_auto.model_type_to_module_name(model_type)
            module_name = f"transformers.models.{module_name}"
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(module_name)
        return getattr(self._modules[module_name], attr)


MODEL_MAPPING = _LazyAutoMapping(
    configuration_auto.CONFIG_MAPPING_NAMES,
    {
        "bert": "transformers.models.bert.BertForMaskedLM",
        "t5": "coordgen.models.modeling_t5.T5ForConditionalGeneration",
    },
)

MODEL_FOR_COORDINATION_GENERATION_MAPPING = _LazyAutoMapping(
    configuration_auto.CONFIG_MAPPING_NAMES,
    {
        "bert": "coordgen.models.modeling_bert.BertForCoordinationGeneration",
        "t5": "coordgen.models.modeling_t5.T5ForCoordinationGeneration",
    },
)


class _AutoModel(auto_factory._BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


class AutoModelForCoordinationGeneration:
    _model_mapping = MODEL_FOR_COORDINATION_GENERATION_MAPPING
    _auto_model_class = _AutoModel

    def __init__(self, *args, **kwargs):
        class_name = self.__class__.__name__
        raise EnvironmentError(
            f"{class_name} is designed to be instantiated "
            f"using the `{class_name}.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tokenizer = tokenization_auto.AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        model = cls._auto_model_class.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model_class = cls._model_mapping[type(model.config)]
        return model_class(model, tokenizer)
