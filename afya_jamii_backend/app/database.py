import os
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import QueuePool
from app.config import settings
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

print("Effective DATABASE_URL:", settings.DATABASE_URL)
print("Effective DATABASE_PORT:", settings.DATABASE_PORT)

# MySQL connection string with connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    echo=settings.DB_ECHO,
    # MySQL specific options
    connect_args={
        "charset": "utf8mb4",
        "autocommit": False,  # Let SQLAlchemy handle transactions
    }
)

def create_db_and_tables():
    """Create all tables in the database"""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_session():
    """Dependency for getting database session"""
    with Session(engine) as session:
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise e
        finally:
            session.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions (for use outside FastAPI dependencies)"""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise e
    finally:
        session.close()

def test_database_connection():
    """Test database connection"""
    try:
        with Session(engine) as session:
            # Simple query to test connection
            session.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def get_database_stats():
    """Get database connection pool statistics"""
    try:
        with engine.connect() as conn:
            # MySQL specific queries for stats
            result = conn.execute("SHOW STATUS LIKE 'Threads_connected'")
            threads_connected = result.fetchone()
            
            result = conn.execute("SHOW PROCESSLIST")
            process_count = len(result.fetchall())
            
            return {
                "pool_size": engine.pool.size(),
                "checked_out": engine.pool.checkedout(),
                "threads_connected": threads_connected[1] if threads_connected else 0,
                "active_processes": process_count
            }
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}

# Database health check function
def check_database_health():
    """Comprehensive database health check"""
    health_status = {
        "status": "healthy",
        "details": {}
    }
    
    try:
        # Test basic connection
        if not test_database_connection():
            health_status["status"] = "unhealthy"
            health_status["details"]["connection"] = "failed"
            return health_status
        
        # Test table existence
        with Session(engine) as session:
            # Check if essential tables exist
            tables_to_check = ["users", "vitals_records", "conversation_history"]
            for table in tables_to_check:
                try:
                    session.execute(f"SELECT 1 FROM {table} LIMIT 1")
                    health_status["details"][f"table_{table}"] = "exists"
                except Exception:
                    health_status["details"][f"table_{table}"] = "missing"
                    health_status["status"] = "degraded"
        
        # Get connection pool stats
        pool_stats = get_database_stats()
        health_status["details"]["pool_stats"] = pool_stats
        
        # Check for long-running queries (potential issues)
        with Session(engine) as session:
            result = session.execute("""
                SELECT COUNT(*) 
                FROM information_schema.processlist 
                WHERE TIME > 60 AND COMMAND != 'Sleep'
            """)
            long_running = result.scalar()
            if long_running > 0:
                health_status["details"]["long_running_queries"] = long_running
                health_status["status"] = "degraded"
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status

# Database maintenance functions
def vacuum_database():
    """Perform database maintenance (MySQL equivalent)"""
    try:
        with Session(engine) as session:
            # For MySQL, we use OPTIMIZE TABLE instead of VACUUM
            tables = ["users", "vitals_records", "conversation_history"]
            for table in tables:
                session.execute(f"OPTIMIZE TABLE {table}")
            session.commit()
            logger.info("Database maintenance completed successfully")
            return True
    except Exception as e:
        logger.error(f"Database maintenance failed: {e}")
        return False

def backup_database(backup_path: str = "/backups"):
    """Create database backup (skeleton - implement based on your backup strategy)"""
    try:
        # This is a placeholder - implement based on your backup solution
        # Options: mysqldump, Percona XtraBackup, cloud-native solutions, etc.
        logger.info(f"Database backup initiated to {backup_path}")
        # Implement your backup logic here
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return False

# Async session support for future use (if needed)
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    
    # For async operations (future enhancement)
    async_engine = create_async_engine(
        settings.DATABASE_URL.replace("mysql+pymysql", "mysql+aiomysql"),
        echo=settings.DB_ECHO,
        poolclass=QueuePool,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW,
    )

    AsyncSessionLocal = sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async def get_async_session():
        """Async dependency for getting database session"""
        async with AsyncSessionLocal() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Async database session error: {e}")
                raise e
            finally:
                await session.close()

except ImportError:
    # Async dependencies not available
    logger.warning("Async database dependencies not available. Async features disabled.")
    
    async def get_async_session():
        """Fallback async session generator"""
        raise NotImplementedError("Async database sessions not configured")