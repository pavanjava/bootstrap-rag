from llama_deploy import (
    deploy_core,
    ControlPlaneConfig
)
from llama_deploy.message_queues.apache_kafka import KafkaMessageQueueConfig
from dotenv import load_dotenv, find_dotenv
import os


async def main():
    _ = load_dotenv(find_dotenv())

    await deploy_core(
        control_plane_config=ControlPlaneConfig(),
        message_queue_config=KafkaMessageQueueConfig(url=os.environ.get('DEFAULT_KAFKA_URL')),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
