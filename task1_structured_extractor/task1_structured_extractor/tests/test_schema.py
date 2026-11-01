import pytest
from pydantic import ValidationError

from schemas import Invoice


def valid_invoice_data():
    return {
        "invoice_number": "INV-001",
        "customer_name": "Amit Patil",
        "email": "amit@example.com",
        "amount": 2500,
        "due_date": "2026-10-30",
        "payment_status": "Paid",
    }


def test_valid_invoice():
    invoice = Invoice.model_validate(valid_invoice_data())
    assert invoice.invoice_number == "INV-001"
    assert invoice.amount == 2500
    assert invoice.due_date.isoformat() == "2026-10-30"


def test_invalid_email_is_rejected():
    data = valid_invoice_data()
    data["email"] = "not-an-email"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_negative_amount_is_rejected():
    data = valid_invoice_data()
    data["amount"] = -100
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)


def test_unexpected_field_is_rejected():
    data = valid_invoice_data()
    data["extra_field"] = "not allowed"
    with pytest.raises(ValidationError):
        Invoice.model_validate(data)
