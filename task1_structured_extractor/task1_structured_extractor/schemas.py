from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class Invoice(BaseModel):
    """Expected structure for an extracted invoice."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        schema.pop("additionalProperties", None)
        return schema

    invoice_number: str = Field(
        min_length=1,
        description="Unique invoice number exactly as shown in the text.",
    )
    customer_name: str = Field(
        min_length=1,
        description="Customer's full name.",
    )
    email: EmailStr = Field(
        description="Customer email address.",
    )
    amount: float = Field(
        ge=0,
        description="Invoice amount as a number, in INR.",
    )
    due_date: date = Field(
        description="Payment due date in YYYY-MM-DD format.",
    )
    payment_status: Literal["Pending", "Paid", "Overdue", "Unknown"] = Field(
        description="Payment status. Use Unknown if the text does not state it.",
    )
