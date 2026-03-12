from unittest.mock import MagicMock, patch

from app.database.sql_client import get_all_dynamic_for_parking


def test_get_all_dynamic_for_parking_groups_by_type():
    """get_all_dynamic_for_parking should return a nested dict grouped by type."""
    mock_rows = [
        MagicMock(type="price", key="hourly", value="3.50"),
        MagicMock(type="price", key="daily", value="25.00"),
        MagicMock(type="availability", key="total_spaces", value="200"),
    ]
    with patch("app.database.sql_client.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.all.return_value = mock_rows
        result = get_all_dynamic_for_parking("parking_001")
        assert "price" in result
        assert result["price"]["hourly"] == "3.50"
        assert "availability" in result


def test_get_dynamic_value_returns_none_for_missing_key():
    """get_dynamic_value should return None when a key does not exist."""
    with patch("app.database.sql_client.SessionLocal") as mock_session_cls:
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        from app.database.sql_client import get_dynamic_value

        result = get_dynamic_value("parking_001", "nonexistent_key")
        assert result is None
