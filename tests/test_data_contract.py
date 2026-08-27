"""数据契约的最小自动化测试。"""

from __future__ import annotations

import pandas as pd

from scripts.validate_dataset import validate_relationships, validate_table_contract


def valid_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": pd.Series(["p-1", "p-2"], dtype="string"),
            "product_locale": pd.Series(["us", "us"], dtype="string"),
            "product_title": pd.Series(["Lamp", "Chair"], dtype="string"),
            "product_description": pd.Series([pd.NA, "A chair"], dtype="string"),
            "product_bullet_point": pd.Series([pd.NA, "Light"], dtype="string"),
            "product_brand": pd.Series(["Brand A", pd.NA], dtype="string"),
            "product_color": pd.Series(["Black", pd.NA], dtype="string"),
        }
    )


def valid_examples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": pd.Series(["e-1", "e-2"], dtype="string"),
            "query": pd.Series(["desk lamp", "chair"], dtype="string"),
            "query_id": pd.Series(["q-1", "q-2"], dtype="string"),
            "product_id": pd.Series(["p-1", "p-2"], dtype="string"),
            "product_locale": pd.Series(["us", "us"], dtype="string"),
            "esci_label": pd.Series(["E", "S"], dtype="string"),
            "small_version": pd.Series([1, 1], dtype="Int8"),
            "large_version": pd.Series([0, 0], dtype="Int8"),
            "split": pd.Series(["train", "test"], dtype="string"),
        }
    )


def valid_sources() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": pd.Series(["q-1", "q-2"], dtype="string"),
            "source": pd.Series(["source-a", "source-b"], dtype="string"),
        }
    )


def test_valid_contract_has_no_problems() -> None:
    assert validate_table_contract("products", valid_products()) == []
    assert validate_table_contract("examples", valid_examples()) == []
    assert validate_table_contract("sources", valid_sources()) == []
    assert validate_relationships(valid_products(), valid_examples(), valid_sources()) == []


def test_required_field_and_product_id_type_are_blocking() -> None:
    products = valid_products().drop(columns=["product_title"])
    problems = validate_table_contract("products", products)
    assert any(item["check"] == "products.schema.required_columns" for item in problems)

    numeric_products = valid_products()
    numeric_products["product_id"] = [1, 2]
    problems = validate_table_contract("products", numeric_products)
    assert any(item["check"] == "products.product_id.type" for item in problems)


def test_enum_and_duplicate_relation_are_blocking() -> None:
    examples = valid_examples()
    examples.loc[0, "esci_label"] = "X"
    examples.loc[1, "example_id"] = "e-1"
    products = valid_products()
    products.loc[1, "product_id"] = "p-1"

    table_problems = validate_table_contract("examples", examples)
    relation_problems = validate_relationships(products, examples, valid_sources())

    assert any(item["check"] == "examples.esci_label.enum" for item in table_problems)
    assert any(item["check"] == "examples.example_id.unique" for item in relation_problems)
    assert any(item["check"] == "products.logical_primary_key.unique" for item in relation_problems)


def test_broken_composite_foreign_key_is_blocking() -> None:
    examples = valid_examples()
    examples.loc[0, "product_id"] = "not-in-products"
    problems = validate_relationships(valid_products(), examples, valid_sources())
    assert any(item["check"] == "examples.products.referential_integrity" for item in problems)
