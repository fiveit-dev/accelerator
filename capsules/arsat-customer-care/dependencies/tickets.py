from abc import ABC, abstractmethod
import datetime
from typing import List, Optional
from models.ticket import Ticket
import httpx
from loguru import logger
import json
import base64
import xmltodict
import textwrap
from config.manager import settings


class TicketsProvider(ABC):

    @abstractmethod
    async def get_active_tickets(self, customer_id: str) -> List[Ticket]:
        """Get all active tickets for a given customer."""
        ...

    @abstractmethod
    async def get_ticket_by_id(
        self, ticket_id: str, customer_id: str
    ) -> Optional[Ticket]:
        """Retrieve a single ticket by its ID and customer ID."""
        ...


class MaximoTicketsProvider(TicketsProvider):

    def __init__(
        self, base_url, user_id, passwd, request_timeout=10.0, verify_ssl=True
    ):
        # Para autenticar en la parte de Headers (Usuario y contraseña en Base 64)
        token = base64.b64encode(f"{user_id}:{passwd}".encode()).decode()
        self.base_params = {"lean": "1", "_action": "query"}    
        
        self._client = httpx.AsyncClient(
            base_url = base_url,
            headers = {"maxauth": token},
            verify=verify_ssl, 
            timeout=request_timeout
            )

    async def __aenter__(self):
        """Permite el uso con 'async with'."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cierra el cliente al salir del bloque 'async with'."""
        await self._client.aclose()    
        
    def _parse_tickets_from_response(self, response: httpx.Response) -> List[Ticket]:
        
        response.raise_for_status() 
        
        data = xmltodict.parse(response.text)
        tickets_data = (
            data.get("QueryA_TKWL_IA_RHResponse", {})
            .get("A_TKWL_IA_RHSet", {})
            .get("TICKET", [])
        )

        # Maximo puede devolver un solo ticket como un diccionario en lugar de una lista.
        if isinstance(tickets_data, dict):
            tickets_data = [tickets_data]
            
        logger.debug(f"Tickets encontrados: {len(tickets_data)}")
        return [Ticket.from_maximo_xml(t) for t in tickets_data]
    
    async def get_active_tickets(self, customer_id) -> List[Ticket]:

        statuses = ",".join(f"'{s}'" for s in settings.MAXIMO_OPEN_TICKET_STATUSES)
        where_clause = f"(status in ({statuses}) and pluspcustomer='{customer_id}')"

        # prueba de cuerpo XML para la consulta de tickets usando QueryA_TKWL_IA_RH
        xml_body = textwrap.dedent(f"""
            <max:QueryA_TKWL_IA_RH xmlns:max="http://www.ibm.com/maximo">
                <max:A_TKWL_IA_RHQuery operandMode="AND">
                    <max:WHERE>{where_clause}</max:WHERE>
                </max:A_TKWL_IA_RHQuery>
            </max:QueryA_TKWL_IA_RH>
        """)

        logger.debug(f"Attemping to query tickets with {self.base_url}")
        
        response = await self._client.post(
            "/",
            headers={"Content-Type": "application/xml"},
            params=self.base_params,
            content=xml_body.strip()
        )

        return self._parse_tickets_from_response(response)
        
    async def get_ticket_by_id(self, ticket_id, customer_id) -> Optional[Ticket]:
        params = {
            **self.base_params,
            "ticketid": ticket_id,
            "pluspcustomer": customer_id,
        }

        logger.debug(f"Attemping to query tickets with {self.base_url}")

        response = await self._client.get("/", params=params)

        tickets = self._parse_tickets_from_response(response)
        return tickets[0] if tickets else None


class MaximoFakeTicketsProvider(TicketsProvider):

    def __init__(self, tickets_path="mocks/maximo_tickets.json"):
        self.tickets = json.load(open(tickets_path))["QueryA_TKWL_IA_RHResponse"][
            "A_TKWL_IA_RHSet"
        ]["TICKET"]

    async def get_active_tickets(self, customer_id) -> List[Ticket]:
        return [
            Ticket._dump(t)
            for t in self.tickets
            if t["Attributes"]["STATUS"]["content"]
            in settings.MAXIMO_OPEN_TICKET_STATUSES
            and t["Attributes"]["PLUSPCUSTOMER"]["content"] == customer_id
        ]

    async def get_ticket_by_id(self, ticket_id, customer_id) -> Optional[Ticket]:
        tickets = [
            Ticket.from_maximo_dump(t)
            for t in self.tickets
            if t["Attributes"]["TICKETID"]["content"] == ticket_id
            and t["Attributes"]["PLUSPCUSTOMER"]["content"] == customer_id
        ]
        if tickets and len(tickets) > 0:
            return tickets[0]
        return None

def get_tickets_provider():
    def _get_maximo_provider() -> TicketsProvider:
        if settings.MAXIMO_BASE_URL:
            logger.debug("Using MaximoTicketsProvider with real Maximo instance")
            return MaximoTicketsProvider(
                f"{settings.MAXIMO_BASE_URL}/A_TKWL_IA_RH",
                settings.MAXIMO_USER_ID,
                settings.MAXIMO_PASSWD,
                request_timeout=settings.MAXIMO_REQUEST_TIMEOUT,
                verify_ssl=settings.MAXIMO_HTTP_VERIFY_SSL,
            )
        else:
            logger.debug("Using MaximoFakeTicketsProvider with mock data")
            return MaximoFakeTicketsProvider()

    return _get_maximo_provider
