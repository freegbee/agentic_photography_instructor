from typing import Generic, Type, Dict, Any, TypedDict, TypeVar, get_origin, Union, get_args

T = TypeVar("T")


class HyperparameterStore(Generic[T]):
    """Speichert und validiert Hyperparameter für eine `TypedDict`-Klasse.

        Der generische Typ `T` entspricht der übergebenen `typedict_cls`.
        Beim `set`/`update` werden Schlüssel und (einfache) Typen gegen die Annotationen von
        `typedict_cls` geprüft. Ist `typedict_cls.__total__ == True`, sind alle annotierten
        Felder erforderlich. Methoden: `set`, `update`, `get`, `as_dict`, `__repr__`.
    """

    def __init__(self, typedict_cls: Type[T], initial: Dict[str, Any] | None = None):
        self._typedict_cls = typedict_cls
        self._annotations = getattr(typedict_cls, "__annotations__", {})
        self._total = getattr(typedict_cls, "__total__", True)
        self._data: Dict[str, Any] = {}
        if initial:
            self.set(initial)

    def _is_instance_of_expected(self, value: Any, expected: Any) -> bool:
        # einfache Prüfung: wenn expected ein direkter Typ ist, isinstance prüfen,
        # bei Union/Optional nur Basisfall behandeln
        origin = get_origin(expected)
        if origin is None:
            if isinstance(expected, type):
                return isinstance(value, expected)
            return True
        if origin is Union and getattr(__import__("typing"), "Union", None):
            args = get_args(expected)
            return any(self._is_instance_of_expected(value, arg) for arg in args)
        return True

    def _validate_dict(self, d: Dict[str, Any]) -> None:
        # keys gültig?
        unknown = set(d) - set(self._annotations)
        if unknown:
            raise KeyError(f"Unknow hyperparameter: {unknown}")
        # required keys prüfen (bei total=True sind alle annotierten Schlüssel erforderlich)
        if self._total:
            missing = set(self._annotations) - set(d)
            if missing:
                raise KeyError(f"Missing required hyperparameter: {missing}")
        # Typprüfung (einfach)
        for k, v in d.items():
            expected = self._annotations.get(k)
            if expected is not None and not self._is_instance_of_expected(v, expected):
                raise TypeError(f"Wrong type for '{k}': expected {expected}, provided {type(v)}")

    def set(self, d: Dict[str, Any]) -> None:
        self._validate_dict(d)
        self._data = dict(d)

    def update(self, **kwargs: Any) -> None:
        new = dict(self._data)
        new.update(kwargs)
        self._validate_dict(new)
        self._data = new

    def get(self) -> T:
        return dict(self._data)  # return copy

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __repr__(self) -> str:
        return f"<HyperparameterStore {self._typedict_cls.__name__}: {self._data}>"


S = TypeVar("S")


class HyperparameterRegistry:
    _registry: Dict[Type[TypedDict], HyperparameterStore] = {}

    @classmethod
    def get_store(cls, typedict_cls: Type[S], initial: Dict[str, Any] | None = None) -> HyperparameterStore[S]:
        existing = cls._registry.get(typedict_cls)
        if existing is not None:
            # falls initial übergeben wurde, merge/override prüfen
            if initial:
                existing.update(**initial)  # validiert intern
            return existing  # type: ignore[return-value]
        store = HyperparameterStore(typedict_cls, initial=initial)
        cls._registry[typedict_cls] = store
        return store  # type: ignore[return-value]
