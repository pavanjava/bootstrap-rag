import os

from llama_deploy import (
    deploy_core,
    ControlPlaneConfig
)
from llama_deploy.message_queues.rabbitmq import RabbitMQMessageQueueConfig
from dotenv import load_dotenv, find_dotenv


async def main():

    _ = load_dotenv(find_dotenv())

    await deploy_core(
        control_plane_config=ControlPlaneConfig(),
        message_queue_config=RabbitMQMessageQueueConfig(url=os.environ.get('RABBITMQ_DEFAULT_URL'),
                                                        exchange_name=os.environ.get('RABBITMQ_EXCHANGE_NAME')),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())