# from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import HTTPBearer
# from fastapi.responses import JSONResponse
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from typing import List, Optional
# from datetime import datetime
# import json
# import logging
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
# import time

# from app.config import settings
# # from app.models import (
# #     UserResponse, UserCreate, UserLogin, VitalsInput, VitalsSubmission,
# #     CombinedResponse, MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
# # )
# from app.auth import get_current_active_user, authenticate_user, create_access_token, get_password_hash
# from app.ml_model import risk_model, initialize_model
# from app.llm_groq import afya_llm, initialize_llm_service
# from app.database import get_session, create_db_and_tables
# from sqlmodel import Session, select

# from app.models import (
#     UserDB, VitalsRecord, ConversationHistory,   # ← add these three
#     UserResponse, UserCreate, UserLogin, VitalsInput, VitalsSubmission,
#     CombinedResponse, MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
# )


# # Configure logging
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL),
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Rate limiting
# limiter = Limiter(key_func=get_remote_address)

# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Afya Jamii AI - Clinical Decision Support System for Kenyan Maternal Healthcare",
#     version="1.0.0",
#     docs_url="/docs" if settings.DEBUG else None,
#     redoc_url="/redoc" if settings.DEBUG else None
# )

# # Add rate limit exceeded handler
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# # Security middleware
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["127.0.0.1"] if not settings.DEBUG else ["*"]
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE"],
#     allow_headers=["*"],
# )

# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)
#     response.headers["X-Content-Type-Options"] = "nosniff"
#     response.headers["X-Frame-Options"] = "DENY"
#     response.headers["X-XSS-Protection"] = "1; mode=block"
#     response.headers["Content-Security-Policy"] = settings.CSP_DIRECTIVES
#     return response

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
#     return response

# # Initialize components on startup
# @app.on_event("startup")
# async def startup_event():
#     logger.info("Starting Afya Jamii AI application...")
    
#     # Create database tables
#     try:
#         create_db_and_tables()
#         logger.info("Database tables created successfully")
#     except Exception as e:
#         logger.error(f"Database initialization failed: {e}")
#         raise
    
#     # Initialize ML model
#     model_loaded = initialize_model()
#     if not model_loaded:
#         logger.error("ML model failed to load")
#         raise RuntimeError("ML model initialization failed")
#     logger.info("ML model loaded successfully")
    
#     # Initialize LLM service
#     llm_ready = initialize_llm_service()
#     if not llm_ready:
#         logger.warning("LLM service initialization failed - running in limited mode")
#     else:
#         logger.info("LLM service initialized successfully")
    
#     logger.info("Afya Jamii AI application started successfully")

# # Health check endpoint
# @app.get("/", include_in_schema=False)
# async def root():
#     return {"message": "Afya Jamii AI API is running", "status": "healthy"}

# @app.get("/health")
# @limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
# async def health_check(request: Request):
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow(),
#         "version": "1.0.0",
#         "services": {
#             "database": "connected",
#             "ml_model": risk_model.model is not None,
#             "llm_service": afya_llm.llm is not None
#         }
#     }

# # Authentication endpoints
# @app.post("/api/v1/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# @limiter.limit("10/minute")
# async def signup(request: Request, user_data: UserCreate, session: Session = Depends(get_session)):
#     # Check if user exists
#     existing_user = session.exec(select(UserDB).where(
#         (UserDB.username == user_data.username) | (UserDB.email == user_data.email)
#     )).first()
    
#     if existing_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username or email already registered"
#         )
    
#     hashed_password = get_password_hash(user_data.password)
#     db_user = UserDB(
#         username=user_data.username,
#         email=user_data.email,
#         full_name=user_data.full_name,
#         account_type=user_data.account_type,
#         hashed_password=hashed_password
#     )
    
#     session.add(db_user)
#     session.commit()
#     session.refresh(db_user)
    
#     logger.info(f"New user registered: {user_data.username}")
#     return db_user

# @app.post("/api/v1/auth/login", response_model=Token)
# @limiter.limit("5/minute")
# async def login(request: Request, login_data: UserLogin, session: Session = Depends(get_session)):
#     user = authenticate_user(session, login_data.username, login_data.password)
#     if not user:
#         logger.warning(f"Failed login attempt for user: {login_data.username}")
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#         )
    
#     access_token = create_access_token(data={"sub": user.username})
#     logger.info(f"User logged in successfully: {login_data.username}")
#     return Token(
#         access_token=access_token,
#         token_type="bearer",
#         expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
#     )

# # Main vitals submission endpoint
# @app.post("/api/v1/vitals/submit", response_model=CombinedResponse)
# @limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
# async def submit_vitals(
#     request: Request,
#     submission: VitalsSubmission,
#     background_tasks: BackgroundTasks,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         # Prepare features for ML model
#         features = {
#             'Age': submission.vitals.age,
#             'SystolicBP': submission.vitals.systolic_bp,
#             'DiastolicBP': submission.vitals.diastolic_bp,
#             'BS': submission.vitals.bs,
#             'BodyTemp': submission.vitals.body_temp,
#             'HeartRate': submission.vitals.heart_rate
#         }
        
#         # Get ML prediction
#         risk_label, probability, feature_importances = risk_model.predict(features)
        
#         # Store vitals record
#         vitals_record = VitalsRecord(
#             user_id=current_user.id,
#             age=submission.vitals.age,
#             systolic_bp=submission.vitals.systolic_bp,
#             diastolic_bp=submission.vitals.diastolic_bp,
#             bs=submission.vitals.bs,
#             body_temp=submission.vitals.body_temp,
#             body_temp_unit=submission.vitals.body_temp_unit,
#             heart_rate=submission.vitals.heart_rate,
#             patient_history=submission.vitals.patient_history,
#             ml_risk_label=risk_label,
#             ml_probability=probability,
#             ml_feature_importances=json.dumps(feature_importances) if feature_importances else None
#         )
        
#         session.add(vitals_record)
#         session.commit()
#         session.refresh(vitals_record)
        
#         # Prepare ML output
#         ml_output = MLModelOutput(
#             risk_label=risk_label,
#             probability=probability,
#             feature_importances=feature_importances
#         )
        
#         # Generate initial LLM advice
#         llm_prompt_data = {
#             'age': submission.vitals.age,
#             'systolic_bp': submission.vitals.systolic_bp,
#             'diastolic_bp': submission.vitals.diastolic_bp,
#             'bs': submission.vitals.bs,
#             'body_temp': submission.vitals.body_temp,
#             'temp_unit': submission.vitals.body_temp_unit,
#             'heart_rate': submission.vitals.heart_rate,
#             'account_type': submission.account_type.value,
#             'ml_model_output': risk_label,
#             'probability': probability,
#             'feature_importances': feature_importances,
#             'patient_history': submission.vitals.patient_history or "No significant history provided",
#             'question': "Provide initial risk assessment and recommendations based on the vitals data.",
#             'history': ""
#         }
        
#         advice_text = afya_llm.generate_advice(llm_prompt_data)
        
#         llm_advice = LLMAdviceResponse(
#             advice=advice_text,
#             timestamp=datetime.utcnow()
#         )
        
#         # Store initial conversation
#         conversation = ConversationHistory(
#             user_id=current_user.id,
#             vitals_record_id=vitals_record.id,
#             user_message="Initial assessment request",
#             ai_response=advice_text
#         )
        
#         session.add(conversation)
#         session.commit()
        
#         response = CombinedResponse(
#             user_id=current_user.id,
#             submission_id=vitals_record.id,
#             timestamp=datetime.utcnow(),
#             ml_output=ml_output,
#             llm_advice=llm_advice
#         )
        
#         logger.info(f"Vitals submitted for user {current_user.username}, risk: {risk_label}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error processing vitals for user {current_user.username}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error processing health data"
#         )

# # LLM conversation endpoint
# @app.post("/api/v1/chat/advice", response_model=LLMAdviceResponse)
# @limiter.limit("30/minute")
# async def get_llm_advice(
#     request: Request,
#     advice_request: LLMAdviceRequest,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         # Get latest vitals record for context
#         statement = select(VitalsRecord).where(
#             VitalsRecord.user_id == current_user.id
#         ).order_by(VitalsRecord.created_at.desc()).limit(1)
        
#         latest_vitals = session.exec(statement).first()
        
#         if not latest_vitals:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="No health data available. Please submit vitals first."
#             )
        
#         # Prepare LLM prompt data
#         llm_prompt_data = {
#             'age': latest_vitals.age,
#             'systolic_bp': latest_vitals.systolic_bp,
#             'diastolic_bp': latest_vitals.diastolic_bp,
#             'bs': latest_vitals.bs,
#             'body_temp': latest_vitals.body_temp,
#             'temp_unit': latest_vitals.body_temp_unit,
#             'heart_rate': latest_vitals.heart_rate,
#             'account_type': current_user.account_type.value,
#             'ml_model_output': latest_vitals.ml_risk_label,
#             'probability': latest_vitals.ml_probability,
#             'feature_importances': json.loads(latest_vitals.ml_feature_importances) if latest_vitals.ml_feature_importances else {},
#             'patient_history': latest_vitals.patient_history or "No significant history provided",
#             'question': advice_request.question
#         }
        
#         advice_text = afya_llm.generate_advice(llm_prompt_data)
        
#         # Store conversation
#         conversation = ConversationHistory(
#             user_id=current_user.id,
#             vitals_record_id=latest_vitals.id,
#             user_message=advice_request.question,
#             ai_response=advice_text
#         )
        
#         session.add(conversation)
#         session.commit()
        
#         response = LLMAdviceResponse(
#             advice=advice_text,
#             timestamp=datetime.utcnow()
#         )
        
#         logger.info(f"LLM advice generated for user {current_user.username}")
#         return response
        
#     except Exception as e:
#         logger.error(f"Error generating LLM advice for user {current_user.username}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error generating advice"
#         )

# # History endpoints
# @app.get("/api/v1/history/vitals", response_model=List[VitalsRecord])
# @limiter.limit("60/minute")
# async def get_vitals_history(
#     request: Request,
#     limit: int = 10,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     statement = select(VitalsRecord).where(
#         VitalsRecord.user_id == current_user.id
#     ).order_by(VitalsRecord.created_at.desc()).limit(limit)
    
#     records = session.exec(statement).all()
#     return records

# @app.get("/api/v1/history/conversations", response_model=List[ConversationHistory])
# @limiter.limit("60/minute")
# async def get_conversation_history(
#     request: Request,
#     limit: int = 20,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     statement = select(ConversationHistory).where(
#         ConversationHistory.user_id == current_user.id
#     ).order_by(ConversationHistory.created_at.desc()).limit(limit)
    
#     conversations = session.exec(statement).all()
#     return conversations

# # Error handlers
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"detail": exc.detail}
#     )

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.error(f"Unhandled exception: {exc}")
#     return JSONResponse(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         content={"detail": "Internal server error"}
#     )





# app/main.py
# from datetime import datetime
# from typing import List
# import json, logging, time

# from fastapi import (
#     FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
# )
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from fastapi.responses import JSONResponse

# from slowapi import Limiter
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
# from sqlmodel import Session, select

# from app.config import settings
# from app.auth import (
#     get_current_active_user, authenticate_user, create_access_token, get_password_hash
# )
# from app.ml_model import risk_model, initialize_model
# from app.llm_groq import afya_llm, initialize_llm_service
# from app.database import get_session, create_db_and_tables
# from app.models import (
#     UserDB, VitalsRecord, ConversationHistory,
#     UserResponse, UserCreate, UserLogin, VitalsSubmission, CombinedResponse,
#     MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
# )

# # --- Logging ---
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL, "INFO"),
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger("app.main")

# # --- Rate limiting ---
# limiter = Limiter(key_func=get_remote_address)

# # --- App object ---
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Afya Jamii AI - Clinical Decision Support System",
#     version="1.0.0",
#     docs_url="/docs",  # Swagger UI
#     redoc_url="/redoc"  # ReDoc
# )
# app.state.limiter = limiter

# # Custom rate limit handler
# @app.exception_handler(RateLimitExceeded)
# def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
#     return JSONResponse(
#         status_code=429,
#         content={"detail": "Rate limit exceeded. Please try again later."}
#     )

# # --- Middlewares ---
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["*"] if settings.DEBUG else ["127.0.0.1"]
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)
#     response.headers.update({
#         "X-Content-Type-Options": "nosniff",
#         "X-Frame-Options": "DENY",
#         "X-XSS-Protection": "1; mode=block",
#         "Content-Security-Policy": settings.CSP_DIRECTIVES,
#     })
#     return response

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start = time.time()
#     response = await call_next(request)
#     logger.info(f"{request.method} {request.url.path} → {response.status_code} "
#                 f"({time.time()-start:.2f}s)")
#     return response

# # --- Startup tasks ---
# @app.on_event("startup")
# def startup_event():
#     logger.info("Booting Afya Jamii AI app...")
#     try:
#         create_db_and_tables()
#         logger.info("DB tables created/verified.")
#     except Exception as e:
#         logger.error(f"DB init failed: {e}")
#         raise

#     if not initialize_model():
#         raise RuntimeError("ML model init failed")
#     logger.info("ML model loaded.")

#     if initialize_llm_service():
#         logger.info("LLM service ready.")
#     else:
#         logger.warning("LLM init failed → limited mode.")

#     logger.info("Afya Jamii AI started successfully.")

# # --- Health endpoints ---
# @app.get("/", include_in_schema=False)
# async def root():
#     return {"message": "Afya Jamii AI API is running", "status": "healthy"}

# @app.get("/health")
# @limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
# async def health_check(request: Request):
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "version": "1.0.0",
#         "services": {
#             "database": "connected",
#             "ml_model": risk_model.model is not None,
#             "llm_service": afya_llm.llm is not None
#         }
#     }

# # --- Auth endpoints ---
# @app.post("/api/v1/auth/signup", response_model=UserResponse,
#           status_code=status.HTTP_201_CREATED)
# @limiter.limit("10/minute")
# async def signup(request: Request, user_data: UserCreate,
#                  session: Session = Depends(get_session)):
#     existing = session.exec(
#         select(UserDB).where((UserDB.username == user_data.username) |
#                              (UserDB.email == user_data.email))
#     ).first()
#     if existing:
#         raise HTTPException(status_code=400, detail="Username or email already registered")

#     hashed_pw = get_password_hash(user_data.password)
#     db_user = UserDB(**user_data.dict(exclude={"password"}),
#                      hashed_password=hashed_pw)
#     session.add(db_user)
#     session.commit()
#     session.refresh(db_user)
#     logger.info(f"User registered: {user_data.username}")
#     return db_user

# @app.post("/api/v1/auth/login", response_model=Token)
# @limiter.limit("5/minute")
# async def login(request: Request, login_data: UserLogin,
#                 session: Session = Depends(get_session)):
#     user = authenticate_user(session, login_data.username, login_data.password)
#     if not user:
#         raise HTTPException(status_code=401, detail="Incorrect username or password")

#     token = create_access_token(data={"sub": user.username})
#     return Token(access_token=token, token_type="bearer",
#                  expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)

# # --- Vitals submission ---
# @app.post("/api/v1/vitals/submit", response_model=CombinedResponse)
# async def submit_vitals(
#     request: Request,
#     submission: VitalsSubmission,
#     background_tasks: BackgroundTasks,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     features = {
#         'Age': submission.vitals.age,
#         'SystolicBP': submission.vitals.systolic_bp,
#         'DiastolicBP': submission.vitals.diastolic_bp,
#         'BS': submission.vitals.bs,
#         'BodyTemp': submission.vitals.body_temp,
#         'HeartRate': submission.vitals.heart_rate
#     }
#     risk_label, prob, feat_imp = risk_model.predict(features)

#     vitals_record = VitalsRecord(
#         user_id=current_user.id,
#         **submission.vitals.dict(),
#         ml_risk_label=risk_label,
#         ml_probability=prob,
#         ml_feature_importances=json.dumps(feat_imp) if feat_imp else None
#     )
#     session.add(vitals_record)
#     session.commit()
#     session.refresh(vitals_record)

#     ml_output = MLModelOutput(risk_label=risk_label, probability=prob,
#                               feature_importances=feat_imp)
#     llm_prompt_data = {
#         **features,
#         'account_type': submission.account_type.value,
#         'ml_model_output': risk_label,
#         'probability': prob,
#         'feature_importances': feat_imp,
#         'patient_history': submission.vitals.patient_history or "No history",
#         'question': "Provide initial risk assessment.",
#         'history': ""
#     }
#     advice = afya_llm.generate_advice(llm_prompt_data)
#     llm_advice = LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

#     convo = ConversationHistory(
#         user_id=current_user.id, vitals_record_id=vitals_record.id,
#         user_message="Initial assessment request", ai_response=advice
#     )
#     session.add(convo)
#     session.commit()

#     return CombinedResponse(user_id=current_user.id,
#                             submission_id=vitals_record.id,
#                             timestamp=datetime.utcnow(),
#                             ml_output=ml_output,
#                             llm_advice=llm_advice)

# # --- LLM chat ---
# @app.post("/api/v1/chat/advice", response_model=LLMAdviceResponse)
# async def get_llm_advice(
#     request: Request,
#     advice_request: LLMAdviceRequest,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     latest = session.exec(
#         select(VitalsRecord).where(VitalsRecord.user_id == current_user.id)
#         .order_by(VitalsRecord.created_at.desc()).limit(1)
#     ).first()
#     if not latest:
#         raise HTTPException(status_code=400, detail="No health data available.")

#     llm_prompt_data = {
#         'age': latest.age, 'systolic_bp': latest.systolic_bp,
#         'diastolic_bp': latest.diastolic_bp, 'bs': latest.bs,
#         'body_temp': latest.body_temp, 'temp_unit': latest.body_temp_unit,
#         'heart_rate': latest.heart_rate,
#         'account_type': current_user.account_type.value,
#         'ml_model_output': latest.ml_risk_label,
#         'probability': latest.ml_probability,
#         'feature_importances': json.loads(latest.ml_feature_importances or "{}"),
#         'patient_history': latest.patient_history or "No history",
#         'question': advice_request.question
#     }
#     advice = afya_llm.generate_advice(llm_prompt_data)

#     convo = ConversationHistory(user_id=current_user.id,
#                                 vitals_record_id=latest.id,
#                                 user_message=advice_request.question,
#                                 ai_response=advice)
#     session.add(convo)
#     session.commit()

#     return LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

# # --- History endpoints ---
# @app.get("/api/v1/history/vitals", response_model=List[VitalsRecord])
# async def get_vitals_history(request: Request,
#                              limit: int = 10,
#                              current_user: UserDB = Depends(get_current_active_user),
#                              session: Session = Depends(get_session)):
#     return session.exec(
#         select(VitalsRecord)
#         .where(VitalsRecord.user_id == current_user.id)
#         .order_by(VitalsRecord.created_at.desc()).limit(limit)
#     ).all()

# @app.get("/api/v1/history/conversations", response_model=List[ConversationHistory])
# async def get_conversation_history(request: Request,
#                                    limit: int = 20,
#                                    current_user: UserDB = Depends(get_current_active_user),
#                                    session: Session = Depends(get_session)):
#     return session.exec(
#         select(ConversationHistory)
#         .where(ConversationHistory.user_id == current_user.id)
#         .order_by(ConversationHistory.created_at.desc()).limit(limit)
#     ).all()

# # --- Error handlers ---
# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     logger.error(f"{exc.status_code}: {exc.detail}")
#     return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.exception("Unhandled exception")
#     return JSONResponse(status_code=500, content={"detail": "Internal server error"})



# app/main.py
# from datetime import datetime
# from typing import List
# import json, logging, time

# from fastapi import (
#     FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
# )
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from fastapi.responses import JSONResponse

# from slowapi import Limiter
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address
# from sqlmodel import Session, select

# from app.config import settings
# from app.auth import (
#     get_current_active_user, authenticate_user,
#     create_access_token, get_password_hash
# )
# from app.ml_model import risk_model, initialize_model
# from app.llm_groq import afya_llm, initialize_llm_service
# from app.database import get_session, create_db_and_tables
# from app.models import (
#     UserDB, VitalsRecord, ConversationHistory,
#     UserResponse, UserCreate, UserLogin, VitalsSubmission, CombinedResponse,
#     MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
# )

# # ───────────────────── LOGGING ─────────────────────
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL, "INFO"),
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("app.main")

# # ───────────────────── RATE LIMITER ────────────────
# limiter = Limiter(key_func=get_remote_address)

# # ───────────────────── FASTAPI APP ────────────────
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Afya Jamii AI - Clinical Decision Support System",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc"
# )
# app.state.limiter = limiter

# # ───────────────────── MIDDLEWARES ────────────────
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["*"] if settings.DEBUG else ["127.0.0.1"]
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS or ["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     response = await call_next(request)
#     response.headers.update({
#         "X-Content-Type-Options": "nosniff",
#         "X-Frame-Options": "DENY",
#         "X-XSS-Protection": "1; mode=block",
#     })
#     if not settings.DEBUG:
#         response.headers["Content-Security-Policy"] = settings.CSP_DIRECTIVES
#     return response

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start = time.time()
#     response = await call_next(request)
#     logger.info(f"{request.method} {request.url.path} → {response.status_code} "
#                 f"({time.time()-start:.2f}s)")
#     return response

# # ───────────────────── EXCEPTION HANDLERS ─────────
# @app.exception_handler(RateLimitExceeded)
# def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
#     return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     logger.warning(f"{exc.status_code}: {exc.detail}")
#     return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     logger.exception("Unhandled exception")
#     return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# # ───────────────────── STARTUP ─────────────────────
# @app.on_event("startup")
# def startup_event():
#     logger.info("Booting Afya Jamii AI app...")
#     try:
#         create_db_and_tables()
#         logger.info("DB tables created/verified.")
#     except Exception as e:
#         logger.error(f"DB init failed: {e}")
#         raise RuntimeError("DB initialization failed") from e

#     if not initialize_model():
#         raise RuntimeError("ML model init failed")
#     logger.info("ML model loaded.")

#     if not initialize_llm_service():
#         logger.warning("LLM init failed → limited mode.")
#     else:
#         logger.info("LLM service ready.")

#     logger.info("Afya Jamii AI started successfully.")

# # ───────────────────── HEALTH ─────────────────────
# @app.get("/", include_in_schema=False)
# async def root():
#     return {"message": "Afya Jamii AI API is running", "status": "healthy"}

# @app.get("/health")
# @limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
# async def health_check(request: Request):
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat(),
#         "version": "1.0.0",
#         "services": {
#             "database": "connected",
#             "ml_model": bool(risk_model.model),
#             "llm_service": bool(afya_llm.llm)
#         }
#     }

# # ───────────────────── AUTH ─────────────────────
# @app.post("/api/v1/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# @limiter.limit("10/minute")
# async def signup(request: Request, user_data: UserCreate, session: Session = Depends(get_session)):
#     try:
#         existing = session.exec(
#             select(UserDB).where((UserDB.username == user_data.username) | (UserDB.email == user_data.email))
#         ).first()
#         if existing:
#             raise HTTPException(status_code=400, detail="Username or email already registered")

#         hashed_pw = get_password_hash(user_data.password)
#         db_user = UserDB(**user_data.dict(exclude={"password"}), hashed_password=hashed_pw)
#         session.add(db_user)
#         session.commit()
#         session.refresh(db_user)
#         logger.info(f"User registered: {user_data.username}")
#         return db_user
#     except Exception as e:
#         logger.error(f"Signup failed: {e}")
#         raise HTTPException(status_code=500, detail="Could not register user")

# @app.post("/api/v1/auth/login", response_model=Token)
# @limiter.limit("5/minute")
# async def login(request: Request, login_data: UserLogin, session: Session = Depends(get_session)):
#     user = authenticate_user(session, login_data.username, login_data.password)
#     if not user:
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
#     token = create_access_token(data={"sub": user.username})
#     return Token(access_token=token, token_type="bearer",
#                  expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)

# # ───────────────────── VITALS ─────────────────────
# @app.post("/api/v1/vitals/submit", response_model=CombinedResponse)
# async def submit_vitals(
#     request: Request,
#     submission: VitalsSubmission,
#     background_tasks: BackgroundTasks,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         features = {
#             'Age': submission.vitals.age,
#             'SystolicBP': submission.vitals.systolic_bp,
#             'DiastolicBP': submission.vitals.diastolic_bp,
#             'BS': submission.vitals.bs,
#             'BodyTemp': submission.vitals.body_temp,
#             'HeartRate': submission.vitals.heart_rate
#         }
#         risk_label, prob, feat_imp = risk_model.predict(features)

#         vitals_record = VitalsRecord(
#             user_id=current_user.id,
#             **submission.vitals.dict(),
#             ml_risk_label=risk_label,
#             ml_probability=prob,
#             ml_feature_importances=json.dumps(feat_imp) if feat_imp else None
#         )
#         session.add(vitals_record)
#         session.commit()
#         session.refresh(vitals_record)

#         ml_output = MLModelOutput(risk_label=risk_label, probability=prob, feature_importances=feat_imp)

#         llm_prompt_data = {
#             **features,
#             'account_type': submission.account_type.value,
#             'ml_model_output': risk_label,
#             'probability': prob,
#             'feature_importances': feat_imp,
#             'patient_history': submission.vitals.patient_history or "No history",
#             'question': "Provide initial risk assessment.",
#             'history': ""
#         }
#         advice = afya_llm.generate_advice(llm_prompt_data)
#         llm_advice = LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

#         convo = ConversationHistory(
#             user_id=current_user.id, vitals_record_id=vitals_record.id,
#             user_message="Initial assessment request", ai_response=advice
#         )
#         session.add(convo)
#         session.commit()

#         return CombinedResponse(
#             user_id=current_user.id,
#             submission_id=vitals_record.id,
#             timestamp=datetime.utcnow(),
#             ml_output=ml_output,
#             llm_advice=llm_advice
#         )
#     # except Exception as e:
#     #     logger.error(f"Vitals submission failed: {e}")
#     #     raise HTTPException(status_code=500, detail="Vitals submission failed")
    
    
# # ───────────────────── LLM CHAT ─────────────────────
# @app.post("/api/v1/chat/advice", response_model=LLMAdviceResponse)
# async def get_llm_advice(
#     request: Request,
#     advice_request: LLMAdviceRequest,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         latest = session.exec(
#             select(VitalsRecord).where(VitalsRecord.user_id == current_user.id)
#             .order_by(VitalsRecord.created_at.desc()).limit(1)
#         ).first()
#         if not latest:
#             raise HTTPException(status_code=400, detail="No health data available.")

#         llm_prompt_data = {
#             'age': latest.age, 'systolic_bp': latest.systolic_bp,
#             'diastolic_bp': latest.diastolic_bp, 'bs': latest.bs,
#             'body_temp': latest.body_temp, 'temp_unit': latest.body_temp_unit,
#             'heart_rate': latest.heart_rate,
#             'account_type': current_user.account_type.value,
#             'ml_model_output': latest.ml_risk_label,
#             'probability': latest.ml_probability,
#             'feature_importances': json.loads(latest.ml_feature_importances or "{}"),
#             'patient_history': latest.patient_history or "No history",
#             'question': advice_request.question
#         }
#         advice = afya_llm.generate_advice(llm_prompt_data)

#         convo = ConversationHistory(user_id=current_user.id, vitals_record_id=latest.id,
#                                     user_message=advice_request.question, ai_response=advice)
#         session.add(convo)
#         session.commit()

#         return LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())
#     except Exception as e:
#         logger.error(f"LLM advice failed: {e}")
#         raise HTTPException(status_code=500, detail="LLM advice retrieval failed")

# # ───────────────────── HISTORY ─────────────────────
# @app.get("/api/v1/history/vitals", response_model=List[VitalsRecord])
# async def get_vitals_history(request: Request, limit: int = 10,
#                              current_user: UserDB = Depends(get_current_active_user),
#                              session: Session = Depends(get_session)):
#     try:
#         return session.exec(
#             select(VitalsRecord).where(VitalsRecord.user_id == current_user.id)
#             .order_by(VitalsRecord.created_at.desc()).limit(limit)
#         ).all()
#     except Exception as e:
#         logger.error(f"Vitals history fetch failed: {e}")
#         raise HTTPException(status_code=500, detail="Could not fetch vitals history")

# @app.get("/api/v1/history/conversations", response_model=List[ConversationHistory])
# async def get_conversation_history(request: Request, limit: int = 20,
#                                    current_user: UserDB = Depends(get_current_active_user),
#                                    session: Session = Depends(get_session)):
#     try:
#         return session.exec(
#             select(ConversationHistory).where(ConversationHistory.user_id == current_user.id)
#             .order_by(ConversationHistory.created_at.desc()).limit(limit)
#         ).all()
#     except Exception as e:
#         logger.error(f"Conversation history fetch failed: {e}")
#         raise HTTPException(status_code=500, detail="Could not fetch conversation history")



# app/main.py
# from datetime import datetime
# from typing import List
# import json
# import logging
# import time
# import traceback

# from fastapi import (
#     FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
# )
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.trustedhost import TrustedHostMiddleware
# from fastapi.responses import JSONResponse

# from slowapi import Limiter
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address
# from sqlmodel import Session, select

# from app.config import settings
# from app.auth import (
#     get_current_active_user, authenticate_user,
#     create_access_token, get_password_hash
# )
# from app.ml_model import risk_model, initialize_model
# from app.llm_groq import afya_llm, initialize_llm_service
# from app.database import get_session, create_db_and_tables
# from app.models import (
#     UserDB, VitalsRecord, ConversationHistory,
#     UserResponse, UserCreate, UserLogin, VitalsSubmission, CombinedResponse,
#     MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
# )

# # ───────────────────── LOGGING ─────────────────────
# logging.basicConfig(
#     level=getattr(logging, settings.LOG_LEVEL, "INFO"),
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger("app.main")

# # ───────────────────── RATE LIMITER ────────────────
# limiter = Limiter(key_func=get_remote_address)

# # ───────────────────── FASTAPI APP ────────────────
# app = FastAPI(
#     title=settings.PROJECT_NAME,
#     description="Afya Jamii AI - Clinical Decision Support System",
#     version="1.0.0",
#     docs_url="/docs" if settings.DEBUG else "/docs",   # keep docs enabled; CSP handles blocking
#     redoc_url="/redoc" if settings.DEBUG else "/redoc"
# )
# app.state.limiter = limiter

# # ───────────────────── MIDDLEWARES ────────────────
# app.add_middleware(
#     TrustedHostMiddleware,
#     allowed_hosts=["*"] if settings.DEBUG else ["127.0.0.1"]
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.CORS_ORIGINS or ["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.middleware("http")
# async def add_security_headers(request: Request, call_next):
#     # Add core security headers; avoid blocking Swagger in DEBUG
#     response = await call_next(request)
#     response.headers.update({
#         "X-Content-Type-Options": "nosniff",
#         "X-Frame-Options": "DENY",
#         "X-XSS-Protection": "1; mode=block",
#     })
#     if not settings.DEBUG and getattr(settings, "CSP_DIRECTIVES", None):
#         response.headers["Content-Security-Policy"] = settings.CSP_DIRECTIVES
#     return response

# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     start = time.time()
#     try:
#         response = await call_next(request)
#     except Exception as e:
#         # log full stack trace for unexpected middleware/handler failures
#         logger.exception(f"Unhandled exception while processing request {request.method} {request.url.path}")
#         raise
#     duration = time.time() - start
#     logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.3f}s) from {request.client.host}")
#     return response

# # ───────────────────── EXCEPTION HANDLERS ─────────
# @app.exception_handler(RateLimitExceeded)
# def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
#     return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})

# @app.exception_handler(HTTPException)
# async def http_exception_handler(request: Request, exc: HTTPException):
#     logger.warning(f"HTTPException for {request.method} {request.url.path}: {exc.detail}")
#     return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# @app.exception_handler(Exception)
# async def general_exception_handler(request: Request, exc: Exception):
#     # Log full traceback to server log (do NOT expose secrets)
#     logger.exception(f"Unhandled exception for {request.method} {request.url.path}")
#     # Include a short helpful message in the response
#     return JSONResponse(status_code=500, content={"detail": "Internal server error — check server logs for details."})

# # ───────────────────── STARTUP ─────────────────────
# @app.on_event("startup")
# def startup_event():
#     logger.info("Starting Afya Jamii AI app startup sequence...")
#     try:
#         create_db_and_tables()
#         logger.info("Database tables created/verified.")
#     except Exception as e:
#         logger.exception("Database initialization failed")
#         # fail fast — don't start app in inconsistent state
#         raise RuntimeError("Database initialization failed") from e

#     try:
#         ok = initialize_model()
#         if not ok:
#             raise RuntimeError("initialize_model returned falsy")
#         logger.info("ML model loaded.")
#     except Exception:
#         logger.exception("ML model initialization failed")
#         raise RuntimeError("ML model init failed")

#     try:
#         llm_ok = initialize_llm_service()
#         if llm_ok:
#             logger.info("LLM service initialized.")
#         else:
#             logger.warning("LLM initialization returned falsy — running reduced LLM mode")
#     except Exception:
#         logger.exception("LLM initialization raised exception; continuing in limited mode")

#     logger.info("Afya Jamii startup complete.")

# # -------------------- Helpers --------------------
# def _safe_json_loads(s: str):
#     try:
#         return json.loads(s)
#     except Exception:
#         return {}

# # -------------------- Endpoints --------------------

# @app.get("/", include_in_schema=False)
# async def root():
#     return {"message": "Afya Jamii AI API is running", "status": "healthy"}

# @app.get("/health")
# @limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
# async def health_check(request: Request):
#     try:
#         return {
#             "status": "healthy",
#             "timestamp": datetime.utcnow().isoformat(),
#             "version": "1.0.0",
#             "services": {
#                 "database": "connected",
#                 "ml_model": bool(getattr(risk_model, "model", None)),
#                 "llm_service": bool(getattr(afya_llm, "llm", None))
#             }
#         }
#     except Exception:
#         logger.exception("Health check failed")
#         raise HTTPException(status_code=500, detail="Health check failed")

# # ------------ Auth ------------
# @app.post("/api/v1/auth/signup", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# @limiter.limit("10/minute")
# async def signup(request: Request, user_data: UserCreate, session: Session = Depends(get_session)):
#     try:
#         existing = session.exec(
#             select(UserDB).where((UserDB.username == user_data.username) | (UserDB.email == user_data.email))
#         ).first()
#         if existing:
#             raise HTTPException(status_code=400, detail="Username or email already registered")

#         hashed_pw = get_password_hash(user_data.password)
#         db_user = UserDB(**user_data.dict(exclude={"password"}), hashed_password=hashed_pw)
#         session.add(db_user)
#         session.commit()
#         session.refresh(db_user)
#         logger.info("New user created: %s", db_user.username)
#         return db_user
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Signup failed")
#         raise HTTPException(status_code=500, detail="Could not register user")

# @app.post("/api/v1/auth/login", response_model=Token)
# @limiter.limit("5/minute")
# async def login(request: Request, login_data: UserLogin, session: Session = Depends(get_session)):
#     try:
#         user = authenticate_user(session, login_data.username, login_data.password)
#         if not user:
#             raise HTTPException(status_code=401, detail="Incorrect username or password")
#         token = create_access_token(data={"sub": user.username})
#         logger.info("User logged in: %s", user.username)
#         return Token(access_token=token, token_type="bearer",
#                      expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("Login failed")
#         raise HTTPException(status_code=500, detail="Login failed")

# # ------------ Vitals submission ------------
# @app.post("/api/v1/vitals/submit", response_model=CombinedResponse)
# async def submit_vitals(
#     request: Request,
#     submission: VitalsSubmission,
#     background_tasks: BackgroundTasks,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         # log received payload (safe: avoid logging PII)
#         logger.debug("submit_vitals payload: %s", submission.dict())

#         features = {
#             "Age": submission.vitals.age,
#             "SystolicBP": submission.vitals.systolic_bp,
#             "DiastolicBP": submission.vitals.diastolic_bp,
#             "BS": submission.vitals.bs,
#             "BodyTemp": submission.vitals.body_temp,
#             "HeartRate": submission.vitals.heart_rate,
#         }

#         risk_label, prob, feat_imp = risk_model.predict(features)

#         vitals_record = VitalsRecord(
#             user_id=current_user.id,
#             **submission.vitals.dict(),
#             ml_risk_label=risk_label,
#             ml_probability=prob,
#             ml_feature_importances=json.dumps(feat_imp) if feat_imp else None
#         )
#         session.add(vitals_record)
#         session.commit()
#         session.refresh(vitals_record)

#         ml_output = MLModelOutput(
#             risk_label=risk_label,
#             probability=prob,
#             feature_importances=feat_imp
#         )

#         llm_prompt_data = {
#             **features,
#             "account_type": submission.account_type.value,
#             "ml_model_output": risk_label,
#             "probability": prob,
#             "feature_importances": feat_imp,
#             "patient_history": submission.vitals.patient_history or "No history",
#             "question": "Provide initial risk assessment.",
#             "history": ""
#         }

#         try:
#             advice = afya_llm.generate_advice(llm_prompt_data)
#         except Exception:
#             logger.exception("LLM generate_advice failed - continuing without LLM")
#             advice = "LLM currently unavailable; please consult a clinician."

#         llm_advice = LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

#         convo = ConversationHistory(
#             user_id=current_user.id,
#             vitals_record_id=vitals_record.id,
#             user_message="Initial assessment request",
#             ai_response=advice
#         )
#         session.add(convo)
#         session.commit()

#         return CombinedResponse(
#             user_id=current_user.id,
#             submission_id=vitals_record.id,
#             timestamp=datetime.utcnow(),
#             ml_output=ml_output,
#             llm_advice=llm_advice
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         # detailed server log with traceback
#         logger.exception("Vitals submission failed: %s", str(e))
#         # return generic message to client
#         raise HTTPException(status_code=500, detail="Vitals submission failed - see server logs")

# # ------------ LLM Chat ------------
# @app.post("/api/v1/chat/advice", response_model=LLMAdviceResponse)
# async def get_llm_advice(
#     request: Request,
#     advice_request: LLMAdviceRequest,
#     current_user: UserDB = Depends(get_current_active_user),
#     session: Session = Depends(get_session)
# ):
#     try:
#         latest = session.exec(
#             select(VitalsRecord).where(VitalsRecord.user_id == current_user.id)
#             .order_by(VitalsRecord.created_at.desc()).limit(1)
#         ).first()
#         if not latest:
#             raise HTTPException(status_code=400, detail="No health data available.")

#         llm_prompt_data = {
#             "age": latest.age,
#             "systolic_bp": latest.systolic_bp,
#             "diastolic_bp": latest.diastolic_bp,
#             "bs": latest.bs,
#             "body_temp": latest.body_temp,
#             "temp_unit": latest.body_temp_unit,
#             "heart_rate": latest.heart_rate,
#             "account_type": current_user.account_type.value,
#             "ml_model_output": latest.ml_risk_label,
#             "probability": latest.ml_probability,
#             "feature_importances": json.loads(latest.ml_feature_importances or "{}"),
#             "patient_history": latest.patient_history or "No history",
#             "question": advice_request.question
#         }

#         advice = afya_llm.generate_advice(llm_prompt_data)

#         convo = ConversationHistory(
#             user_id=current_user.id,
#             vitals_record_id=latest.id,
#             user_message=advice_request.question,
#             ai_response=advice
#         )
#         session.add(convo)
#         session.commit()

#         return LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

#     except HTTPException:
#         raise
#     except Exception:
#         logger.exception("LLM advice retrieval failed")
#         raise HTTPException(status_code=500, detail="LLM advice retrieval failed")

# # ------------ History ------------
# @app.get("/api/v1/history/vitals", response_model=List[VitalsRecord])
# async def get_vitals_history(request: Request, limit: int = 10,
#                              current_user: UserDB = Depends(get_current_active_user),
#                              session: Session = Depends(get_session)):
#     try:
#         records = session.exec(
#             select(VitalsRecord).where(VitalsRecord.user_id == current_user.id)
#             .order_by(VitalsRecord.created_at.desc()).limit(limit)
#         ).all()
#         return records
#     except Exception:
#         logger.exception("Vitals history fetch failed")
#         raise HTTPException(status_code=500, detail="Could not fetch vitals history")

# @app.get("/api/v1/history/conversations", response_model=List[ConversationHistory])
# async def get_conversation_history(request: Request, limit: int = 20,
#                                    current_user: UserDB = Depends(get_current_active_user),
#                                    session: Session = Depends(get_session)):
#     try:
#         convos = session.exec(
#             select(ConversationHistory).where(ConversationHistory.user_id == current_user.id)
#             .order_by(ConversationHistory.created_at.desc()).limit(limit)
#         ).all()
#         return convos
#     except Exception:
#         logger.exception("Conversation history fetch failed")
#         raise HTTPException(status_code=500, detail="Could not fetch conversation history")


# app/main.py (relevant changes)

import json
import numpy as np
import logging
from datetime import datetime
from fastapi import (
    FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
)
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.models import (
    UserDB, VitalsRecord, ConversationHistory,
    UserResponse, UserCreate, UserLogin, VitalsSubmission, CombinedResponse,
    MLModelOutput, LLMAdviceRequest, LLMAdviceResponse, Token
)
from app.ml_model import risk_model
from app.llm_groq import afya_llm
from app.auth import (
    get_current_active_user, authenticate_user,
    create_access_token, get_password_hash
)
from app.database import get_session

# ------------------------------------------------------
# ✅ 1. Create FastAPI instance BEFORE route definitions
# ------------------------------------------------------
logger = logging.getLogger("app.main")
app = FastAPI(title="AfyaJamii API", version="1.0.0")


# ------------- JSON Safe Converter -------------
def json_safe(o):
    """Convert NumPy types & arrays to Python-native for JSON serialization."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o


# ------------ Vitals submission ------------
@app.post("/api/v1/vitals/submit", response_model=CombinedResponse)
async def submit_vitals(
    request: Request,
    submission: VitalsSubmission,
    background_tasks: BackgroundTasks,
    current_user: UserDB = Depends(get_current_active_user),
    session: Session = Depends(get_session)
):
    """
    Accept vitals from authenticated user, run ML risk model,
    generate LLM advice, store record in DB, and return combined response.
    """
    try:
        logger.debug("submit_vitals payload: %s", submission.dict())

        features = {
            "Age": submission.vitals.age,
            "SystolicBP": submission.vitals.systolic_bp,
            "DiastolicBP": submission.vitals.diastolic_bp,
            "BS": submission.vitals.bs,
            "BodyTemp": submission.vitals.body_temp,
            "HeartRate": submission.vitals.heart_rate,
        }

        # Run ML model & force Python-native types
        raw_label, raw_prob, raw_feat_imp = risk_model.predict(features)
        risk_label = str(raw_label)
        prob = float(raw_prob)  # ensure float
        feat_imp = json_safe(raw_feat_imp)  # ensure safe for JSON

        vitals_record = VitalsRecord(
            user_id=current_user.id,
            **submission.vitals.dict(),
            ml_risk_label=risk_label,
            ml_probability=prob,
            ml_feature_importances=json.dumps(feat_imp, default=json_safe) if feat_imp else None
        )
        session.add(vitals_record)
        session.commit()
        session.refresh(vitals_record)

        ml_output = MLModelOutput(
            risk_label=risk_label,
            probability=prob,
            feature_importances=feat_imp
        )

        llm_prompt_data = {
            **features,
            "account_type": submission.account_type.value,
            "ml_model_output": risk_label,
            "probability": prob,
            "feature_importances": feat_imp,
            "patient_history": submission.vitals.patient_history or "No history",
            "question": "Provide initial risk assessment.",
            "history": ""
        }

        try:
            advice = afya_llm.generate_advice(llm_prompt_data)
        except Exception:
            logger.exception("LLM generate_advice failed - continuing without LLM")
            advice = "LLM currently unavailable; please consult a clinician."

        llm_advice = LLMAdviceResponse(advice=advice, timestamp=datetime.utcnow())

        convo = ConversationHistory(
            user_id=current_user.id,
            vitals_record_id=vitals_record.id,
            user_message="Initial assessment request",
            ai_response=advice
        )
        session.add(convo)
        session.commit()

        return CombinedResponse(
            user_id=current_user.id,
            submission_id=vitals_record.id,
            timestamp=datetime.utcnow(),
            ml_output=ml_output,
            llm_advice=llm_advice
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Vitals submission failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Vitals submission failed - see server logs")