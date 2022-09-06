#from sklearn import utils
from ModelExplainer import globalExplainerPostProcess, localExplainerPostProcess, localGlobalOverLap
import narrationreviewsutils
from parsedbreads import *
from itertools import islice
from annotator_questions import local_questions, global_questions, compare_questions, combineList, question_preamble,\
    eval_metric_questions, compare_preamble_questions, global_questions_categorical, datasetInfoPicker
from AnnotationInstance import *
from collections import OrderedDict
from proj_db.database import engine, sessionLocal
from proj_db import schema
from proj_db.crud import *
import pandas as pd
import traceback
from typing import List, Optional
from fastapi import FastAPI, Request, templating, Depends, status, Cookie, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi_login import LoginManager
from fastapi_login.exceptions import InvalidCredentialsException, HTTPException
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import string
from supertokens_fastapi import supertokens_session, Session as Super_Token_Session
from supertokens_fastapi import create_new_session
from fastapi.staticfiles import StaticFiles
import sys
import os

from proj_db import crud
from pydantic import BaseSettings
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

import secrets
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.utils import get_openapi
import EmailServices
import SimpleEmailer
from fastapi.responses import FileResponse
import tempfile
from proj_db.appConfig import *
from proj_db.db_tables import NarrationsReview
from redeem_code_generator import *
from questions_picker import *
import narrationreviewsutils
import jinja2

import utils
env = jinja2.Environment()
env.globals.update(zip=zip)

fg = 90
task_mapper = {'performance': 'APC', 'local': 'ALC',
               'global': 'AGC', 'compare': 'ACC'}

# Acceptable extension codes for the performance narrative generation
acceptable_extension_codes = ['ZP1T', 'ZX2W', 'ZF4U', 'ZU3F', ]
acceptable_extension_codes_global = ['GP9T', 'ZB9C']


class Settings(BaseSettings):
    openapi_url: str = "/openapi.json"


settings = Settings()
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TaskIdentifier = schema.TaskIdentifier
TaskListReq = schema.TaskListReq
# Set the max_display for the plots
MAX_DISPLAY = MAX_DISPLAY

# Set the maximum number of features to include in the divisions
MAX_DIV_FEATURES = 1000


# db_tables.Base.metadata.create_all(bind=engine)
async def not_found(request: Request, exc):
    return templates.TemplateResponse('page_not_found.html', {
        'request': request, })

exceptions = {
    404: not_found,
}

# openapi_url=settings.openapi_url) (docs_url=None, redoc_url=None)
app = FastAPI(docs_url=None, redoc_url=None, exception_handlers=exceptions)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
security = HTTPBasic()

# Define a simple encryption-Decryption interface
data_encdec = utils.EncyptDecryptData()


@app.on_event("startup")
def setup():
    # print("Creating db tables...")
    db_tables.Base.metadata.create_all(bind=engine)
    # print(f"Created {len(engine.table_names())} tables: {engine.table_names()}")


def get_db():
    try:
        db = sessionLocal()
        yield db
    finally:
        db.close()


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = templating.Jinja2Templates(directory='simplifiedtemplate')
mode_map = {'pending': 0, 'approved': 1, 'all': -1, 'rejected': 2}
# Set up a secret Key
SECRET = 'dummy'

# manager = LoginManager(SECRET, tokenUrl='/auth/login', use_cookie=True)
# manager.cookie_name = 'annotation_instance'
# manager.useRequest(app)


def getAnnotatorInfo(request: Request):
    return {'ip': request.client.host,
            'port': request.client.port, }


def checkisLogin(request: Request, force_template=True):
    try:
        user = json.loads(request.cookies.get('currAnnotate'))

        return user
    except:
        if force_template:
            return templates.TemplateResponse('login_page.html', {
                'request': request, 'vstatus': -1, 'Message': 'Session Expired. Please login again.'})
        else:
            return None


@app.get('/download-narratives/{narrative_type}/{narrative_status}')
async def downloadNarratives(narrative_type: str, narrative_status: str, request: Request, db: Session = Depends(get_db)):
    mode = mode_map[narrative_status]

    user = checkisLogin(request)
    if not (isinstance(user, dict) and user.get('id', None)):
        return user
    if narrative_type not in ["performance", "local", "global", "compare", "all_types"]:
        raise HTTPException(
            status_code=404, detail='Invalid url')
    tempdir = tempfile.mkdtemp()
    saved_umask = os.umask(0o077)
    path = os.path.join(tempdir)

    if narrative_type == 'performance':
        db_narratives = getAllPerformanceNarratives(
            db, narrative_status=int(mode), with_status=True)
    elif narrative_type == 'local':
        db_narratives = getAllLocalNarratives(
            db, narrative_status=int(mode), with_status=True)
    elif narrative_type == 'global':
        db_narratives = getAllGlobalNarratives(
            db, narrative_status=int(mode), with_status=True)
    elif narrative_type == 'compare':
        db_narratives = getAllCompareNarratives(
            db, narrative_status=int(mode), with_status=True)
    else:
        db_narratives_eval = getAllPerformanceNarratives(
            db, narrative_status=int(mode), with_status=True)
        db_narratives_local = getAllLocalNarratives(
            db, narrative_status=int(mode), with_status=True)
        db_narratives_global = getAllGlobalNarratives(
            db, narrative_status=int(mode), with_status=True)
        db_narratives_compare = getAllCompareNarratives(
            db, narrative_status=int(mode), with_status=True)

        db_narratives = {'Performance': db_narratives_eval, 'Global': db_narratives_global,
                         'Compare': db_narratives_compare,
                         'Local': db_narratives_local}

    # print(db_narratives)

    tfile = tempfile.NamedTemporaryFile(
        prefix=f"{narrative_type}_", suffix="_narratives.json", mode="w+", delete=False)
    original_path = tfile.name
    # tfile.name = f'{narrative_type}_narratives.json'
    results = {"data": jsonable_encoder(db_narratives)}
    json.dump(results, tfile)
    tfile.flush()
    # print(tfile.name)

    return FileResponse(path=tfile.name, filename=tfile.name, media_type='text/json')


# Admin only access api page
@app.get("/exclusive")
async def authenticateDocs(request: Request, userID: Optional[str] = Cookie(None),):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/exclusive/ui")
async def loginUI(request: Request, userID: Optional[str] = Cookie(None),):
    annotationInstance = AnnotationSession()
    user = checkisLogin(request)
    # print(user['email'])
    try:

        if isinstance(user, dict) and user.get('id', None):
            response_url = '/dashboard'
            if user['role'] in ['admin', 'super']:
                response_url = '/adminboard'
            resp = RedirectResponse(
                url=response_url, status_code=status.HTTP_302_FOUND)
            return resp
        else:
            return templates.TemplateResponse('login_page.html', {
                'request': request, 'vstatus': 0, 'Message': ''
            })
    except:
        return templates.TemplateResponse('login_page.html', {
            'request': request, 'vstatus': 0, 'Message': ''
        })


@app.post('/change_user_access')
async def change_user_access(request: Request, user_status: schema.UserStatus, db: Session = Depends(get_db)):
    user = crud.changeUserStatus(db, user_status)
    users_narratives = parseMultipleUsersAnnotationsTable(db)
    return users_narratives


@app.get('/adminboard')
async def admindashboard(request: Request, db: Session = Depends(get_db)):
    # annotationInstance = AnnotationSession()
    user = checkisLogin(request,)
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin', 'super']:
            num_annotators = crud.getAnnotators(db)
            annotationInstance = AnnotationSession()
            trq = TaskListReq(task_type='cls')
            tasks = clf_tasks = loadTasks(
                annotationInstance, trq, return_only_count=False)
            nb_narratives = crud.getAllAnnotations(db,)

            # Process a table to display on the admin dashboard showing the statistics of all
            # Narratives per task
            tasks_list = clf_tasks['task_list']
            np.random.shuffle(tasks_list)
            narrative_summaries = []
            task_names = ['randomized']
            for task in tasks_list:
                task_type = 'Classification' if task in clf_tasks['task_list'] else 'Regression'
                task_name = task['t_name']
                task_names.append(task_name)
                # print(task_name)
                task_narrative_summary = getAnnotationStatisticsPerTask(
                    db,  task_name=task_name)

                row = f'''
                <tr>
                <td><a href="/adminboard/{task_type}/task/{task_name}" class="summary_task_name">{task_name}</a></td>
                <!--<td>{task_type}</td>-->
                <td style="text-align:center;">{task_narrative_summary['performance']}</td>
                <td style="text-align:center;">{task_narrative_summary['global']}</td>
                <td style="text-align:center;">{task_narrative_summary['local']}</td>
                <td style="text-align:center;">{task_narrative_summary['Compare']}</td>
                <td style="text-align:center;">{task_narrative_summary['performance']+ task_narrative_summary['global']+ task_narrative_summary['Compare']+task_narrative_summary['local']}</td>
                </tr>
                '''.strip().replace('\n', ' ')
                narrative_summaries.append(row)

            # Load all issues raised by the annotators
            narrative_issues = parseAnnotationIssues(db)

            # Load the users along with summary of their annotations
            users_narratives = parseMultipleUsersAnnotationsTable(db)
            annotation_statistics = getAllAnnotationWorkSummary(db)

            return templates.TemplateResponse('admin_dashboard.html', {'request': request,
                                                                       'user_id': -1,
                                                                       'task_names': task_names,
                                                                       'annotation_statistics': annotation_statistics,
                                                                       'role': user['role'],
                                                                       'nb_tasks': tasks['nb_tasks'],
                                                                       'nb_narratives': nb_narratives,
                                                                       'narrative_summaries': narrative_summaries,
                                                                       'nb_users': len(num_annotators),
                                                                       'narrative_issues':  narrative_issues,
                                                                       'users_narratives': users_narratives})
        else:
            raise HTTPException(status_code=404, detail='Invalid url')
            # annotation_stats = getAnnotationStatisticsForUser(db, int(user['id']))
            # return templates.TemplateResponse(USER_DASHBOARD if int(user['id']) != 2 else 'focusedDashboard.html', {'request': request,
            # 'nav_links':navigation_links,
            #                                                                                                        'role': user['role'],
            #                                                                                                        'annotation_stats': np.sum(list(annotation_stats.values()))})
    else:
        return templates.TemplateResponse('login_page.html', {'request': request,
                                          'vstatus': -1, 'Message': 'Session Expired. Please login again.'})


@app.get('/adminboard/{task_type}/task/{task_name}')
async def serveTaskSpeficipage(task_type: str, task_name: str, request: Request, db: Session = Depends(get_db)):
    user = checkisLogin(request)
    narrative_summaries = ''
    if not (isinstance(user, dict) and user.get('id', None)):
        return user
    # getTaskAnnotationSummary(
    annotation_statistics, annotation_summary_info = getTaskAnnotationSummary(
        db, task_name)
    narrative_summaries = parseNarrativesForTask(annotation_summary_info)
    return templates.TemplateResponse('admin_dashboard_taskstats.html', {
        'user_id': -1,
        'role': user['role'],
        'request': request, 'narrative_summaries': narrative_summaries, 'task_names': [task_name],
        'annotation_statistics': annotation_statistics,
        'current_task_name': task_name
        # 'task_objects': task_objects if task_objects else None,

    })

# Returns the template defining the annotation page for the selected task


@app.get('/adminboard/userProfile/{user_id}')
async def serveUserNarrativesPage(user_id: str,  request: Request, db: Session = Depends(get_db)):
    user = checkisLogin(request)
    narrative_summaries = ''
    if not (isinstance(user, dict) and user.get('id', None)):
        return user
    elif 'BU' not in user_id:
        raise HTTPException(
            status_code=404, detail='Error: No page found')
    else:

        annotationInstance = AnnotationSession()
        narrative_summaries, task_objects = parseAllNarrativesForAnnotator(
            db, annotationInstance, user_id.replace('BU', ''))
        task_names = [t['t_name'] for t in task_objects]
        # {% set model_names = task_object.modelIdx.keys() %}
        annotation_statistics = getAnnotatorWorkSummary(
            db, user_id.replace('BU', ''))
        user_info = getUserInfobyId(db, user_id.replace('BU', ''))

    return templates.TemplateResponse('admin_dashboard_userstats.html', {
        'user_id': user_id.replace('BU', ''),
        'role': user['role'],
        'request': request, 'narrative_summaries': narrative_summaries, 'task_names': task_names,
        'annotation_statistics': annotation_statistics, 'user_info': user_info})


@app.post('/get_task_models')
async def getTaskModels(task_name: TaskIdentifier, request: Request):
    models = parsegetTaskModels(task_name.task_name)
    return models


@app.post('/get_task_classes')
async def getTaskClasses(task_name: TaskIdentifier, request: Request):
    classes = parsegetTaskClasses(task_name.task_name)
    return classes


@app.post("/get_task_list")
async def get_task_list(request: Request, task_type: TaskListReq):
    annotationInstance = AnnotationSession()
    tasks = loadTasks(annotationInstance, task_type)

    # print(tasks)
    # random.shuffle(tasks)

    # print(tasks)
    return tasks


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


@ app.get("/docs_")
async def authenticateDocs(request: Request, userID: Optional[str] = Cookie(None),):
    annotationInstance = AnnotationSession()
    user = checkisLogin(request)

    # print(user['email'])
    if isinstance(user, dict) and user.get('id', None):
        app.openapi = custom_openapi
        return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")

    else:
        return templates.TemplateResponse('login_page.html', {
            'request': request, 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
        })


@ app.get('/annotator_logout')
async def logout(request: Request, response: Response):
    request.cookies['currAnnotate'] = {}
    response = templates.TemplateResponse('login_page.html', {
        'request': request, 'vstatus': -1, 'Message': 'Logout successful. You can now close the tab.'
    })
    response.delete_cookie(key='currAnnotate')

    # user = checkisLogin(request)
    response = RedirectResponse(
        url='/exclusive/ui', status_code=status.HTTP_302_FOUND)

    # Delete all user cookies
    response.delete_cookie(key='currAnnotate')
    # response.delete_cookie(key = manager.cookie_name)
    return response


@ app.post('/email_login', response_model=schema.User)
async def getUser(user: schema.User, db: Session = Depends(get_db)):
    return getUserInfo(db, user.email)


@ app.post('/resolve_issue')
async def resolveIssue(issue: schema.ResolvedIssue,  request: Request, db: Session = Depends(get_db)):
    issue = crud.resolveIssue(db, issue_id=issue.issue_id)
    return parseAnnotationIssues(db)


@app.post('/get_annotation_rewards_admin_with_code')
async def searchAnnotationWithCode(rewardCode: schema.PaymentRewardCode, request: Request, db: Session = Depends(get_db)):
    user = checkisLogin(request)
    claim_html = f'''<span>No qualifying points submitted </span>'''
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin',
                            'super']:
            # /get_annotation_rewards_admin
            claims = getClaimWithCode(db, rewardCode.redeem_code)
            claim_html = f'''<span>No qualifying points submitted </span>'''
            if len(claims) > 0:
                claim_html = processAnnotationClaimsForAdmin(claims, db)
            # print([claims[ii].claim_code for ii in range(len(claims))])
            return claim_html

    return claim_html


@app.post('/get_annotation_rewards_admin')
async def getAnnotatorRewardPage(annotation_type: schema.AnnotationReviewRequests, request: Request,  db: Session = Depends(get_db)):

    user = checkisLogin(request,)
    claim_html = f'''<span>No qualifying points submitted </span>'''
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin', 'super']:
            claims = getAllClaims(
                db, annotation_type=annotation_type.annotation_type,
                annotation_status=annotation_type.annotation_status)

            claim_html = f'''<span>No qualifying points submitted </span>'''
            if len(claims) > 0:
                claim_html = processAnnotationClaimsForAdmin(claims, db)
            # print([claims[ii].claim_code for ii in range(len(claims))])
            return claim_html

    return claim_html
# Record Payment
payment_review_mode = {'pay': 1,
                       'reject': 404}


@app.post('/approve_annotator_payment')
async def rewardPayment(payment: schema.AnnotationPaymentReq, request: Request,  db: Session = Depends(get_db)):
    user = checkisLogin(request,)
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin', 'super']:

            res = crud.recordPaidClaims(
                db, payment, payment_mode=payment_review_mode[payment.reviewMode])
            claims = getAllClaims(db, annotation_type=payment.annotation_type,
                                  annotation_status=payment.annotation_status)

            claim_html = f'''<span>No qualifying points submitted </span>'''
            if len(claims) > 0:
                claim_html = processAnnotationClaimsForAdmin(claims, db)
            # print([claims[ii].claim_code for ii in range(len(claims))])
            return claim_html


@app.post('/approve_annotator_payment_extended')
async def rewardPayment(payment: schema.AnnotationPaymentReq, request: Request,  db: Session = Depends(get_db)):
    user = checkisLogin(request,)
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin', 'super']:
            res = crud.recordPaidClaims(
                db, payment, payment_mode=payment_review_mode[payment.reviewMode])
            claims = getClaimsViaCode(db, payment.redeem_code)
            response_url = '/adminboard'
            return RedirectResponse(
                url=response_url, status_code=status.HTTP_302_FOUND)


@app.get('/P-narratorEvaluations')
async def serveNarrationEvaluationPage(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse('performance_narration_evaluation.html', {'request': request,
                                                                                })


@app.post('/narration_quality_load')
async def getNarrationQualityCheck(req: schema.PredictionRequest, db: Session = Depends(get_db)):
    # init the evaluation object
    evaluation_object = narrationreviewsutils.NarrationReviews()
    item = evaluation_object.getRandomItem(req.user_id, db)
    print(item)
    return item


# Admin page for approving payment
@app.get('/claim/{narrative_type}/{rewardCode}/{code_id}')
async def serveRewardPage(narrative_type: str, rewardCode: str, code_id: int, request: Request, db: Session = Depends(get_db)):
    user = checkisLogin(request,)
    claim_html = f'''<span>No qualifying points submitted </span>'''
    narr_html = ''
    if isinstance(user, dict) and user.get('id', None):
        user_id = user.get('id', None)
        if user['role'] in ['admin', 'super']:
            claims = getClaimsViaCode(db, rewardCode)
            annotation_type = claims[0].annotation_type
            annotation_type = claims[0].annotation_type
            annotation_idx = json.loads(claims[0].annotation_entries)
            narratives = getNarrativesForCode(
                db, annotation_idx, annotation_type)
            if narrative_type == 'Performance':
                annotation_type = claims[0].annotation_type
                annotation_idx = json.loads(claims[0].annotation_entries)
                narratives = getNarrativesForCode(
                    db, annotation_idx, annotation_type)
                narr_html = parseperformanceNarrativesSlide(
                    narratives, mode=-1, for_admin=True, for_approval=True, claim_code=rewardCode)
                return templates.TemplateResponse('payment_pages/performance.html', {'request': request,
                                                                                     'code_id': int(code_id),
                                                                                     'narrative_type': narrative_type.lower(),
                                                                                     'narr_html': narr_html,
                                                                                     'narr_type': narrative_type
                                                                                     })
            elif narrative_type == "Local":
                narr_html = parseApprovalLocalNarrativesSlides(
                    narratives, mode=-1, for_admin=True, for_approval=True, claim_code=rewardCode)
            elif narrative_type == "Global":
                narr_html = parseApprovalGlobalNarrativesSlides(
                    narratives, mode=-1, for_admin=True, for_approval=True, claim_code=rewardCode)
            elif narrative_type == "Compare":
                narr_html = parseApprovalCompareNarrativesSlides(
                    narratives, mode=-1, for_admin=True, for_approval=True, claim_code=rewardCode)
                return templates.TemplateResponse('payment_pages/performance.html', {'request': request,
                                                                                     'code_id': int(code_id),
                                                                                     'narrative_type': narrative_type.lower(),
                                                                                     'narr_html': narr_html['slides'],
                                                                                     'fig_plotly_local': narr_html['plotly_inputs_local'],
                                                                                     'fig_plotly_global': narr_html.get('plotly_inputs_global', ''),
                                                                                     'narr_type': narrative_type
                                                                                     })

            return templates.TemplateResponse('payment_pages/performance.html', {'request': request,
                                                                                 'code_id': int(code_id),
                                                                                 'narrative_type': narrative_type.lower(),
                                                                                 'narr_html': narr_html['slides'],
                                                                                 'fig_plotly': narr_html['plotly_inputs'],
                                                                                 'narr_type': narrative_type
                                                                                 })
            pass


# Change the annotation status from admin window


@app.post('/change_compare_narrative_status')
async def changeCompareNarrativeStatus(narrative: schema.AdminNarrativeStatusChange, request: Request,
                                       db: Session = Depends(get_db)):
    mode = mode = narrative.view_mode
    annotationInstance = AnnotationSession()
    narrative_obj = maskSavedCompareNarrative(db, user_id=narrative.user_id, narr_id=narrative.narrative_id,
                                              reason=narrative.status)

    if narrative.task_name == 'rand':
        return {'reload': 'Yes'}

    # Reload all narratives submitted by the user for the task
    isValid, taskObject = annotationInstance.isValidTask(narrative.task_name)

    if int(narrative.user_id) > -1:
        narratives = getCompareNarrativesForUserUsingTaskName(db, int(narrative.user_id),
                                                              task_name=narrative.task_name, narrative_status=mode, with_status=True)
    else:
        narratives = getTaskCompareNarratives(
            db, task_name=narrative.task_name, narrative_status=mode, with_status=True)
    # print(narratives)
    narrative_pack = parseCompareNarrativesSlides(
        narratives, taskObject, mode=mode, for_admin=True)
    # narratives = getPerformanceNarrativesForUser(db, int(narrative.user_id),task_name=narrative.task_name, narrative_status=narrative.view_mode, with_status=True)

    # print(narrative_pack)
    if narratives:
        return narrative_pack
    else:
        # raise HTTPException(
        #    status_code=404, detail='Error: No narrative saved')
        return f'''<div class="dropdown-divider"></div><h4 style="margin-left:20px;">No Records found</h4>'''


@app.post('/change_local_narrative_status')
async def changeLocalNarrativeStatus(narrative: schema.AdminNarrativeStatusChange, request: Request,
                                     db: Session = Depends(get_db)):
    mode = mode = narrative.view_mode
    annotationInstance = AnnotationSession()
    narrative_obj = maskSavedLocalNarrative(db, user_id=narrative.user_id, narr_id=narrative.narrative_id,
                                            reason=narrative.status)

    if narrative.task_name == 'rand':
        return {'reload': 'Yes'}

    # Reload all narratives submitted by the user for the task
    isValid, taskObject = annotationInstance.isValidTask(narrative.task_name)
    if int(narrative.user_id) > -1:
        narratives = getLocalNarrativesForUserUsingTaskName(db, int(narrative.user_id),
                                                            task_name=narrative.task_name, narrative_status=mode, with_status=True)
    else:
        narratives = getTaskLocalNarratives(
            db, task_name=narrative.task_name, narrative_status=mode, with_status=True)
    # print(narratives)
    narrative_pack = parseLocalNarrativesSlides(
        narratives, taskObject, mode=mode, for_admin=True)
    # narratives = getPerformanceNarrativesForUser(db, int(narrative.user_id),task_name=narrative.task_name, narrative_status=narrative.view_mode, with_status=True)

    # print(narrative_pack)
    if narratives:
        return narrative_pack
    else:
        # raise HTTPException(
        #    status_code=404, detail='Error: No narrative saved')
        return f'''<div class="dropdown-divider"></div><h4 style="margin-left:20px;">No Records found</h4>'''


@app.post('/change_global_narrative_status')
async def changeGlobalNarrativeStatus(narrative: schema.AdminNarrativeStatusChange, request: Request,
                                      db: Session = Depends(get_db)):
    mode = mode = narrative.view_mode
    annotationInstance = AnnotationSession()
    narrative_obj = maskSavedGlobalNarrative(db, user_id=narrative.user_id, narr_id=narrative.narrative_id,
                                             reason=narrative.status)
    if narrative.task_name == 'rand':
        return {'reload': 'Yes'}

    # Reload all narratives submitted by the user for the task
    isValid, taskObject = annotationInstance.isValidTask(narrative.task_name)
    if int(narrative.user_id) > -1:
        narratives = getGlobalNarrativesForUserUsingTaskName(db, int(narrative.user_id),
                                                             task_name=narrative.task_name, narrative_status=mode, with_status=True)
    else:
        narratives = getTaskGlobalNarratives(
            db, task_name=narrative.task_name, narrative_status=mode, with_status=True)
    narrative_pack = parseGlobalNarrativesSlides(
        narratives, taskObject, mode=mode, for_admin=True)
    # narratives = getPerformanceNarrativesForUser(db, int(narrative.user_id),task_name=narrative.task_name, narrative_status=narrative.view_mode, with_status=True)

    if narratives:
        return narrative_pack
    else:
        # raise HTTPException(
        #    status_code=404, detail='Error: No narrative saved')
        return f'''<div class="dropdown-divider"></div><h4 style="margin-left:20px;">No Records found</h4>'''


@app.post('/change_performance_narrative_status')
async def changePerformanceNarrativeStatus(narrative: schema.AdminNarrativeStatusChange, request: Request,
                                           db: Session = Depends(get_db)):
    narrative_obj = maskSavedPerformanceNarrative(db,
                                                  user_id=narrative.user_id, narr_id=narrative.narrative_id,
                                                  reason=narrative.status)
    if narrative.task_name == 'rand':
        return {'reload': 'Yes'}
    if int(narrative.user_id) > -1:
        narratives = getPerformanceNarrativesForUserUsingTaskName(db, int(narrative.user_id),
                                                                  task_name=narrative.task_name, narrative_status=narrative.view_mode,
                                                                  with_status=True)
    else:
        narratives = getTaskPerformanceNarratives(db, task_name=narrative.task_name, narrative_status=narrative.view_mode,
                                                  with_status=True)

    if narratives:
        return parseperformanceNarrativesSlide(narratives, mode=narrative.view_mode, for_admin=True)
    else:
        # raise HTTPException(
        #    status_code=404, detail='Error: No narrative saved')
        return f'''<div class="dropdown-divider"></div><h4 style="margin-left:20px;">No Records found</h4>'''


# Get the annotations to approve or reject from the admin window
@app.post('/get_performance_annotations_admin')
async def get_performance_annotations_status(narratives: schema.GetPerformanceNarratives,
                                             request: Request,
                                             db: Session = Depends(get_db)):

    mode = mode_map[narratives.mode]
    if int(narratives.user_id) > -1:
        db_narratives = getPerformanceNarrativesForUserUsingTaskName(db, int(narratives.user_id),
                                                                     task_name=narratives.task_name, narrative_status=mode, with_status=True)
    else:
        db_narratives = getTaskPerformanceNarratives(
            db, narratives.task_name, narrative_status=mode, with_status=True)
    if db_narratives:
        return parseperformanceNarrativesSlide(db_narratives, mode=mode, for_admin=True)
        # return ParsePerformanceNarratives(db_narratives, mode=mode, for_admin=True)
    else:
        # raise HTTPException(
        #    status_code=404, detail='Error: No narrative saved')
        return f'''<div class="dropdown-divider"></div><h4 style="margin-left:20px;">No Records found</h4>'''


@app.post('/get_compare_annotations_admin')
async def get_comparison_annotation_status(narratives: schema.GetStoredNarratives,
                                           request: Request,
                                           db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(narratives.task_name)
    mode = mode_map[narratives.mode]
    if int(narratives.user_id) > -1:
        db_narratives = getCompareNarrativesForUserUsingTaskName(db, int(narratives.user_id),
                                                                 task_name=narratives.task_name, narrative_status=mode, with_status=True)
    else:
        db_narratives = getTaskCompareNarratives(
            db, narratives.task_name,
            narrative_status=mode,
            with_status=True)
    narrative_pack = parseCompareNarrativesSlides(
        db_narratives, taskObject, mode=mode, for_admin=True)
    # print(narrative_pack)
    return narrative_pack


@app.post('/get_local_annotations_admin')
async def get_local_annotation_status(narratives: schema.GetStoredNarratives,
                                      request: Request,
                                      db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(narratives.task_name)
    mode = mode_map[narratives.mode]
    if int(narratives.user_id) > -1:
        db_narratives = getLocalNarrativesForUserUsingTaskName(db, int(narratives.user_id),
                                                               task_name=narratives.task_name, narrative_status=mode, with_status=True)
    else:
        db_narratives = getTaskLocalNarratives(
            db, narratives.task_name, narrative_status=mode, with_status=True)
    narrative_pack = parseLocalNarrativesSlides(
        db_narratives, taskObject, mode=mode, for_admin=True)
    # print(narrative_pack)
    return narrative_pack


@app.post('/get_global_annotations_admin')
async def get_global_annotation_status(narratives: schema.GetStoredNarratives,
                                       request: Request,
                                       db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(narratives.task_name)
    mode = mode_map[narratives.mode]
    if int(narratives.user_id) > -1:
        db_narratives = getGlobalNarrativesForUserUsingTaskName(db, int(narratives.user_id),
                                                                task_name=narratives.task_name, narrative_status=mode, with_status=True)
    else:
        db_narratives = getTaskGlobalNarratives(
            db, task_name=narratives.task_name, narrative_status=mode, with_status=True)
    print([db_narratives[ii].id for ii in range(len(db_narratives))])
    narrative_pack = parseGlobalNarrativesSlides(
        db_narratives, taskObject, mode=mode, for_admin=True)
    # print(narrative_pack)
    return narrative_pack

# @manager.user_loader


def load_user(user_name: str, db: Session = Depends(get_db)):
    user = getUserInfo(db, user_name)
    return user


@app.post('/auth/verify')
async def verfiyUserAccount(request: Request, email: str = Form(...), verfication_code: str = Form(...), db: Session = Depends(get_db)):

    # print(email)

    data = schema.AccountVerification(
        email=email, verification_code=verfication_code)
    user_email = data_encdec.decrypt(data.email)

    # First check if the user is indeed registered
    user = load_user(user_email, db,)
    if not user:
        return {'status': 'Error', 'Message': 'Invalid account'}
    if user.verification_code == data.verification_code:
        # user.is_verified = True

        # Save the changes
        # print(data.email)
        data.email = data_encdec.decrypt(data.email)
        verfiedAccount = crud.verifyAccount(db, data)

        if verfiedAccount:
            # Send an email to confirm verfication
            client = SimpleEmailer.GmailService()
            sign_up_successful = f'''
            Thank you for signing up. You can now proceed with the data annotation.
            '''
            vv = client.sendEmail(user.email, 'Welcome',
                                  message=sign_up_successful)
            mail_sent = ''
            if vv > -1:
                mail_sent = 'Please verify your email address'

            # Login the user
            # access_token = manager.create_access_token(data={'sub': user_email, 'uid': user.id})
            current_user = schema.User(
                id=user.id, email=user.email, role=user.role)
            current_user.is_active = user.is_active
            response_url = '/dashboard'
            if current_user.role in ['admin', 'super']:
                response_url = '/adminboard'
            resp = RedirectResponse(
                url=response_url, status_code=status.HTTP_302_FOUND)
            resp.set_cookie(key='currAnnotate',
                            value=f'{current_user.json()}', )

            # manager.set_cookie(resp, access_token)
            return resp
        else:
            resp = RedirectResponse(
                url=f'/verifyAccount?q={user_email}&vs=-3', status_code=status.HTTP_302_FOUND)
            return resp
            # return {'status':'Error','Message':'An error occured. Please try again later.'}
    else:
        e_enc = data_encdec.encryptData(data.email).decode()
        resp = RedirectResponse(
            url=f'/verifyAccount?q={user_email}&vs=-1', status_code=status.HTTP_302_FOUND)
        return resp


@app.get('/verifyAccount')
async def getAccountVerified(request: Request, q: str, vs: Optional[str] = 0):
    vs = int(vs)
    e_enc = data_encdec.encryptData(q).decode()
    message = ''
    if vs < 0:
        if vs == -1:
            message = 'Invalid verfication code.'
        elif vs == -3:
            message = 'An error occured. Please try again later.'
    return templates.TemplateResponse('account_verification.html', {'request': request, 'vstatus': int(vs), 'Message': message, 'email': str(e_enc)})


@app.post('/auth/login')
async def login(request: Request, data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):

    user_name = data.username
    # password = data.password
    user = load_user(user_name, db,)

    if not user:
        # raise InvalidCredentialsException
        return templates.TemplateResponse('login_page.html', {
            'request': request, 'vstatus': -1, 'Message': 'Account not found. Please create a new account.'
        })
    # elif password != user['password']:
    #    raise InvalidCredentialsException

    if not user.is_verified:
        e_enc = data_encdec.encryptData(user.email).decode()
        # print(str(e_enc))
        return templates.TemplateResponse('account_verification.html', {'request': request, 'vstatus': -1, 'Message': 'Account not verified', 'email': str(e_enc)})
        return {'status': 'Error', 'Message': 'Account not verified'}

    # access_token = manager.create_access_token(data={'sub': user_name, 'uid': user.id})
    current_user = schema.User(id=user.id, email=user.email, role=user.role)
    current_user.is_active = user.is_active
    response_url = '/dashboard'
    if current_user.role in ['admin', 'super']:
        response_url = '/adminboard'
    resp = RedirectResponse(
        url=response_url, status_code=status.HTTP_302_FOUND)
    resp.set_cookie(key='currAnnotate', value=f'{current_user.json()}',

                    )

    # manager.set_cookie(resp, access_token)

    return resp


# Operations for getting all the narratives for a given task from api requests
@app.get('/get_all_global_narratives', response_model=List[schema.SavedGlobalNarration])
async def get_all_globals(db: Session = Depends(get_db)):
    narratives = getAllGlobalNarratives(db)
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.post('/get_task_global_narratives', response_model=List[schema.SavedGlobalNarration])
async def get_Task_GlobalNarrative(task_id: TaskIdentifier, db: Session = Depends(get_db)):
    narratives = getTaskGlobalNarratives(db=db, task_name=task_id.task_name)
    # print(narratives)
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.post('/get_task_local_narratives', response_model=List[schema.SavedLocalNarration])
async def get_Task_LocalNarrative(task_id: TaskIdentifier, db: Session = Depends(get_db)):
    narratives = getTaskLocalNarratives(db=db, task_name=task_id.task_name)
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.get('/get_task_local_narratives', response_model=List[schema.SavedLocalNarration])
async def get_all_LocalNarrative(db: Session = Depends(get_db)):
    narratives = getAllLocalNarratives(db=db,)
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.post('/get_all_task_eval_narratives', response_model=List[schema.SavedEvalNarrations])
async def get_Task_EvalNarrative(db: Session = Depends(get_db)):
    narratives = getAllPerformanceNarratives(db=db, )
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.post('/get_all_task_compare_narratives', response_model=List[schema.SavedCompareNarration])
async def get_CompareNarrative(db: Session = Depends(get_db)):
    narratives = getAllCompareNarratives(db=db, )
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.post('/get_task_compare_narratives', response_model=List[schema.SavedCompareNarration])
async def get_Task_CompareNarrative(task_id: TaskIdentifier, db: Session = Depends(get_db)):
    narratives = getTaskCompareNarratives(db=db, task_name=task_id.task_name)
    if narratives:
        return narratives
    else:
        raise HTTPException(
            status_code=404, detail='Error: No narrative saved')


@app.get('/annotation_guidelines')
async def getGuidelines(request: Request, db: Session = Depends(get_db)):
    user = checkisLogin(request)
    if isinstance(user, dict) and user.get('id', None):
        annotation_stats = getAnnotationStatisticsForUser(db, int(user['id']))
        # print(np.sum(list(annotation_stats.values())))
        return templates.TemplateResponse('annotation_guide.html', {'request': request,
                                                                    'annotation_stats': np.sum(list(annotation_stats.values()))})
    else:
        return user


# Get the UI for the annotator reward page


# Code for the Comparative
@ app.get('/CompareNarratives')
async def loadCompareNarrativesPage(request: Request, db: Session = Depends(get_db)):
    # nb_models = validateURLCode(model_count_code.upper(),acceptable_extension_codes)
    # client_host = request.client.host
    # print(client_host)
    # if nb_models is None:
    #    return templates.TemplateResponse('page_not_found.html', {
    #    'request': request,})
    client_info = getAnnotatorInfo(request)
    randomly_generate = np.random.choice([1, 2, 2, 1])
    annotationInstance = AnnotationSession()
    task_name = 'random'
    model_name = 'random'
    if randomly_generate > 1:
        # choose any of the tasks and choose any of the associated models
        task_list = list(annotationInstance.tasks.tasks_info.keys())
        task_name = random.choice(task_list)
        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        # print(annotationInstance.tasks.tasks_info.keys())
    return templates.TemplateResponse('compareNarratives.html', {
        'request': request,
        'is_random': randomly_generate,
        'task_name': task_name,
        'model_name': model_name,
        'nb_models': 'Zip'.upper(),
        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })

# Code for the Local Narratives


@ app.get('/LocalNarratives')
async def loadLocalNarrativesPage(request: Request, db: Session = Depends(get_db)):
    # nb_models = validateURLCode(model_count_code.upper(),acceptable_extension_codes)
    # client_host = request.client.host
    # print(client_host)
    # if nb_models is None:
    #    return templates.TemplateResponse('page_not_found.html', {
    #    'request': request,})
    client_info = getAnnotatorInfo(request)
    randomly_generate = np.random.choice([1, 2, 2, 1])
    annotationInstance = AnnotationSession()
    task_name = 'random'
    model_name = 'random'
    if randomly_generate > 1:
        # choose any of the tasks and choose any of the associated models
        task_list = list(annotationInstance.tasks.tasks_info.keys())
        task_name = random.choice(task_list)
        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        # print(annotationInstance.tasks.tasks_info.keys())
    return templates.TemplateResponse('localNarratives.html', {
        'request': request,
        'is_random': randomly_generate,
        'task_name': task_name,
        'model_name': model_name,
        'nb_models': 'Zip'.upper(),
        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })

# Code for showing the Global Narratives


@ app.get('/GlobalNarratives')
async def loadGlobalNarrativesPage(request: Request, db: Session = Depends(get_db)):
    # nb_models = validateURLCode(model_count_code.upper(),acceptable_extension_codes)
    # client_host = request.client.host
    # print(client_host)
    # if nb_models is None:
    #    return templates.TemplateResponse('page_not_found.html', {
    #    'request': request,})
    client_info = getAnnotatorInfo(request)
    randomly_generate = np.random.choice([1, 2, 2, 1])
    annotationInstance = AnnotationSession()
    task_name = 'random'
    model_name = 'random'
    if randomly_generate > 1:
        # choose any of the tasks and choose any of the associated models
        task_list = list(annotationInstance.tasks.tasks_info.keys())
        task_name = random.choice(task_list)
        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        # print(annotationInstance.tasks.tasks_info.keys())
    return templates.TemplateResponse('globalNarratives.html', {
        'request': request,
        'is_random': randomly_generate,
        'task_name': task_name,
        'model_name': model_name,
        'nb_models': 'Zip'.upper(),
        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })


# The codes below deals with the performance annotation work
@ app.get('/PerformanceNarratives/{model_count_code}')
async def loadPerformanceNarrativeAnalysisPage(request: Request, model_count_code: str, db: Session = Depends(get_db)):
    nb_models =utils.validateURLCode(model_count_code.upper(),
                                acceptable_extension_codes)
    client_host = request.client.host
    # print(client_host)
    if nb_models is None:
        return templates.TemplateResponse('page_not_found.html', {
            'request': request, })
    randomly_generate = np.random.choice([1, 2, 2, 1])
    annotationInstance = AnnotationSession()
    task_name = 'random'
    model_name = 'random'
    if randomly_generate > 1:
        # choose any of the tasks and choose any of the associated models
        task_list = list(annotationInstance.tasks.tasks_info.keys())
        task_name = random.choice(task_list)

        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        # print(annotationInstance.tasks.tasks_info.keys())
    return templates.TemplateResponse('performanceNarratives.html', {
        'request': request,
        'is_random': randomly_generate,
        'task_name': task_name,
        'model_name': model_name,
        'nb_models': model_count_code.upper(),
        'model_count_code': model_count_code.upper()
        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })


@ app.get('/PerformanceNarratives')
async def loadPerformanceNarrativeAnalysisPage(request: Request,
                                               db: Session = Depends(get_db)):
    randomly_generate = np.random.choice([1, 2, 2, 1])
    annotationInstance = AnnotationSession()
    task_name = 'random'
    model_name = 'random'
    if randomly_generate > 1:
        # choose any of the tasks and choose any of the associated models
        task_list = list(annotationInstance.tasks.tasks_info.keys())
        task_name = random.choice(task_list)
        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        # print(annotationInstance.tasks.tasks_info.keys())
    return templates.TemplateResponse('performanceNarratives.html', {
        'request': request,
        'is_random': randomly_generate,
        'task_name': task_name,
        'model_name': model_name,

        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })


def generateDataSetinfo(nb_classes):
    proportions = []
    # The dataset has 50.% of the data belonging to class C1 and 50.% belonging to class C2
    if nb_classes == 2:
        pos_class = np.round(np.random.uniform(0.3, 0.51), 2)
        # random.shuffle(pos_class)
        proportions = [(1-pos_class), pos_class]
    elif nb_classes == 3:

        pos_class = np.round(np.random.uniform(0.3, 0.35, size=(2)), 2)
        random.shuffle(pos_class)
        proportions = [(1-pos_class.sum()), pos_class[0], pos_class[1]]
    else:
        pos_class = np.round(np.random.uniform(0.2, 0.31, size=(3)), 2)
        random.shuffle(pos_class)
        proportions = [(1-pos_class.sum()), pos_class[0],
                       pos_class[1], pos_class[2]]

    # Build the placeholder maps
    placeholders = {}
    for idx, pr in enumerate(proportions):
        placeholders[f'<p{idx+1}>'] = f'{round(100*pr,2)}%'
        placeholders[f'<c{idx+1}>'] = f'C{idx+1}'

    dataset_info = datasetInfoPicker(nb_classes, placeholders)
    return dataset_info, proportions


def generateRandomPerformanceMetrics():

    annotationInstance = AnnotationSession()

    # Choose the number of labels
    nb_classes = np.random.choice([2, 3, 4])

    # The number of models to show to the user
    nb_models = np.random.choice([1, 1, 2, 3, 4])
    # nb_models=1
    dataset_info, proportions = generateDataSetinfo(nb_classes)

    ignore_more_than_2 = ['auc', 'auc-score']
    evalQuestion = ''

    # randomly generate an evaluation metric table
    evalMetrics = utils.getMetric(nb_models) if nb_models > 1 else [
        utils.getMetric(nb_models)]
    metriclist = list(evalMetrics[0].keys())
    # if nb_classes>2

    edf = pd.DataFrame([list(e.values()) for e in evalMetrics])
    metriclist = list(evalMetrics[0].keys())
    edf.columns = metriclist
    mids = ['A', 'B', 'C', 'D', 'E']
    models_idx = [f'Model {mids[i]}' for i in range(nb_models)]
    edf.index = models_idx
    metric_info = f'''<input id="eval_metric_info" type="hidden" value="{json.dumps(edf.to_json())}"/>
    <input id="task_name_metric" type="hidden" value="randomized"/>
    <input id="slt_model_perform_metric" type="hidden" value="randomized"/>
    '''
    # print(edf)

    evalQuestion = composeQuestions(
        edf, nb_models, evalMetrics)+metric_info+metric_info

    metrics_definitions = utils.getMetricInformation(metriclist)
    return {'metric_table': edf.to_html(classes="table table-striped table-hover eval_tb"),
            'metric_values': json.dumps(edf.to_json()),
            'metric_info': '',
            'evalQuestion': evalQuestion,
            'dataset_info': dataset_info,
            'data_proportions': [],
            'metrics_definitions': metrics_definitions
            }


def getModelMetrics(taskObject, model_name):
    actual_model = taskObject.modelIdx[model_name]
    model_details = taskObject[actual_model]
    # print(model_name)
    evalMetrics = {k: v for k, v in model_details['eval_metrics'].items() if k not in [
        'confusion_matrix', 'cm']}
    return evalMetrics


def composeQuestions(metrics_df, nb_models, evalMetrics):
    models_idx = metrics_df.index.to_list()
    metriclist = metrics_df.columns.to_list()

    # eval_metrics =
    metric_mapped_placeholders = {}
    metric_picks = random.sample(metriclist, k=random.choice(
        [2, 3,  3])) if len(metriclist) > 2 else metriclist
    model_picks = random.sample(models_idx, k=random.choice(
        [2, 2, 3,  3])) if len(models_idx) > 2 else models_idx
    metric_pick, metric_pick2 = random.sample(metriclist, 2)
    # metric_mapped_placeholders = {}
    # print(metric_picks)
    metric_mapped_placeholders['<rand_metrics>'] = combineList(
        metric_picks, ignore_rule=True)
    # print(metric_picks)
    compare_models = random.sample(models_idx, k=2) if len(
        models_idx) > 2 else models_idx

    compare_models2 = random.sample(models_idx, k=2) if len(
        models_idx) > 2 else models_idx

    for i, m in enumerate(metric_picks):
        metric_mapped_placeholders[f'<metric_{1+i}>'] = m
        v = evalMetrics[0][m]
        metric_mapped_placeholders[f'<metric_{1+i}v>'] = str(v)

    metric_mapped_placeholders = {'<rand_model_name1>': compare_models[-1],
                                  '<rand_model_name2>': compare_models[0], '<rand_model_name3>': compare_models2[-1],
                                  '<rand_model_name4>': compare_models2[0], '<model_names>': combineList(models_idx, ignore_rule=True),
                                  '<rand_model_names>': combineList(model_picks, ignore_rule=True),
                                  '<rand_metric>': f'{ metric_pick}',
                                  '<rand_metric1>': f'{metric_pick2}', '<metrics>': combineList(metriclist, ignore_rule=True)}
    metric_mapped_placeholders['<rand_metrics>'] = combineList(
        metric_picks, ignore_rule=True)
    metric_mapped_placeholders['<model_name>'] = compare_models[0]
    try:
        score_1 = metrics_df.loc[compare_models[-1], metric_pick]
        metric_mapped_placeholders['<metric_1_score>'] = str(score_1)

        score_2 = metrics_df.loc[compare_models[0], metric_pick]
        metric_mapped_placeholders['<metric_2_score>'] = str(score_2)

        score_3 = metrics_df.loc[compare_models2[-1], metric_pick2]
        metric_mapped_placeholders['<metric_3_score>'] = str(score_3)

        score_4 = metrics_df.loc[compare_models2[0], metric_pick2]
        metric_mapped_placeholders['<metric_4_score>'] = str(score_4)

        # print(score_1, metric_pick, score_2)
    except:
        pass

    if nb_models == 1:
        evalQuestion = eval_metric_questions_1(
            mapped_placeholders=metric_mapped_placeholders)
    elif nb_models == 2:
        evalQuestion = eval_metric_questions_2(
            mapped_placeholders=metric_mapped_placeholders)
    else:
        evalQuestion = eval_metric_questions_3(
            mapped_placeholders=metric_mapped_placeholders)
    evalQuestion = generateInputBoxesForQuestions(evalQuestion) + \
        f''' <input type="hidden" id="eval_metric_questions" value="{evalQuestion}"/>'''
    return evalQuestion

# Get the prediction for an instance for the local explanation

ignore_tasks =['Air Quality Prediction']
@app.get("/prediction_page")
async def performpredictionss(request: Request,  db: Session = Depends(get_db)):
    # For each task, get statistics on annotations submited by the user
    annotationInstance = AnnotationSession()
    task_lists =[n for n in  list(annotationInstance.tasks.tasks_info.keys()) if n not in ignore_tasks]
    return templates.TemplateResponse('predictions.html', {'request': request,'task_names':task_lists})
@app.post("/load_models_tests")
async def performpredictionss(request: schema.ModelsLoad,):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(request.task_name)
    model_names = list(taskObject.modelIdx.keys())
    models_str = f'<select id="model_names"  class="form-control"><option value="">Choose a model </option>'
    for m in model_names:
        m = taskObject.modelIdx[m]
        models_str += f'<option value="{m}">{m}</option>'
    models_str += f'</select>'
    dataset_x, data_y = taskObject.dataset['test_pack']
    test_idxs = np.arange(len(dataset_x))
    examples_str = f'<select id="test_example" class="form-control"><option value="">Choose an example </option>'
    for ti in test_idxs:
        examples_str += f'<option value="{ti}">test-{ti}</option>'
    examples_str += f'</select>'
    return {'models': models_str,'examples':examples_str}
@app.get('/predictions/{task_name}/{model_name}/{test_id}')
async def predictionAPIGateway(task_name: str, model_name: str, test_id: int, request: Request,):
    test_pred= schema.PredictionRequest(task_name = task_name,model_name = model_name,test_instance = test_id,user_id='Base')
    test_pred.task_name = task_name
    test_pred.model_name = model_name
    test_pred.test_instance = test_id
    results = utils.performPrediction(test_pred)
    return results
@app.post('/simple_prediction')
async def getPrediction(test_pred: schema.PredictionRequest, ):
    results = utils.performPrediction(test_pred)
    return results

@app.post("/perform_prediction")
async def getPrediction(test_pred: schema.PredictionRequest, db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    
    task_lists =[n for n in  list(annotationInstance.tasks.tasks_info.keys()) if n not in ignore_tasks]
    # Randomly select a task for the annotator.
    # The annotator is not shown thesame instance twice
    if test_pred.is_randomize == 1:
        task_name = pickMLTask(
           task_lists , db, problem_type='local')

        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        test_pred.model_name = model_name
        
        dataset_x, data_y = taskObject.dataset['test_pack']
        test_pred.test_instance = np.random.choice(np.arange(len(dataset_x)))
        task_name = pickMLTask(
               task_lists, db, problem_type='local')
        # Make sure the annotator is not attempting the annotation twice.
        alreadySubmitted = checkAlreadyAnnotatedLocal(db, task_name, model_name,
                                                      narrator=test_pred.user_id, test_instance=int(test_pred.test_instance))
        checked = 0
        test_pred.task_name = task_name
        while alreadySubmitted:
            if checked > 8:
                break
            task_name = pickMLTask(
                task_lists, db, problem_type='local')

            isValid, taskObject = annotationInstance.isValidTask(task_name)
            model_name = random.choice(list(taskObject.modelIdx.keys()))
            test_pred.model_name = model_name
            test_pred.task_name = task_name
            dataset_x, data_y = taskObject.dataset['test_pack']
            test_pred.test_instance = np.random.choice(
                np.arange(len(dataset_x)))
            alreadySubmitted = checkAlreadyAnnotatedLocal(db, task_name, model_name,
                                                          narrator=test_pred.user_id, test_instance=int(test_pred.test_instance))
            if not alreadySubmitted:
                break
            else:
                checked += 1
        test_pred.task_name = task_name
    else:
        isValid, taskObject = annotationInstance.isValidTask(
            test_pred.task_name)

    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}

    # if valid model
    actual_model = taskObject.modelIdx[test_pred.model_name]
    test_pred.model_name = actual_model
    model_details = taskObject[actual_model]
    current_model = model_details['model']
    if isinstance(current_model, NeuralModelInterface):
        current_model.loadModel()
    task_type = taskObject.task_type
    if current_model:
        # For now we will read in the global explanation for each classification
        FCC = taskObject.modelsGlobalExplainer[test_pred.model_name]
        test_instance = int(test_pred.test_instance)
        # Get the data instance
        dataset_x, data_y = taskObject.dataset['test_pack']

        x_test = pd.DataFrame([dataset_x[test_instance]],
                              columns=taskObject.task_features.getFeaturesNames())
        correct_output = data_y[test_instance]
        prediction = taskObject.task_features.model_predict(
            current_model, x_test)
        prediction = prediction[0]
        confidence = 0.0
        if task_type == 'cls':
            prediction = int(prediction)
            class_probs = taskObject.task_features.model_predict_proba(
                current_model, x_test)[0]
            # print(prediction,'=== ',class_probs)
            confidence = class_probs[int(prediction)]
            class_probabilities = [f'{c}: {p*100:.2f}%' for c, p in zip(
                list(taskObject.classes_placeholders.keys()), class_probs)]
            # print('Prediction Likelihood: ', ' '.join(class_probabilities))

        # print(dataset_x[test_instance])

        def convVal(vx):
            try:
                vx = float(vx)
            except:
                pass
            return vx
        feature_dict_vals = {fp: convVal(v) for fp, v in zip(taskObject.task_features.getFeaturesPlaceHolderNames(),
                                                             dataset_x[test_instance])}

        # print(feature_dict_vals, ' XXX')
        feature_dict_vals_html = taskObject.task_features.getFeatureInfoAsHtml(
            values=feature_dict_vals, scope_id=f'tp_{test_instance+1}', max_display_per_row=4)

        try:
            ann = dx = FCC.xframe[prediction] if task_type != 'reg' else FCC.xframe[0]

            # feature_names = FCC.feature_names
            nb_features = len(taskObject.task_features.getFeaturesNames())

            # Get the local explanation using Lime
            local_exp = taskObject.modelsLocalExplainer[test_pred.model_name]
            if isinstance(local_exp, LRPExplainerInterface):
                local_exp = LRPExplainerInterface(
                    taskObject.task_features, current_model.model)
                instance, pred = local_exp.explain_data_row(
                    [dataset_x[test_instance]],)
            else:
                instance = local_exp.explain_data_row([dataset_x[test_instance]],
                                                      model=current_model,
                                                      labels=list(
                    range(len(taskObject.task_classes))),
                    num_features=20)
            # processAttribution(instance.local_exp,nb_features,)[prediction]
            attrs = att = instance[prediction] if task_type != 'reg' else instance[1]
            # print(attrs)
            k2, local_rank_list, [local_contradicting,
                                  local_supportive,
                                  local_ignore] = FCC.splitAttribution(attrs,
                                                                       MAX_DIV_FEATURES=MAX_DIV_FEATURES,
                                                                       feature_names=taskObject.task_features.getFeaturesNames())

            k2 = localExplainerPostProcess(
                taskObject.task_features, k2, dataset_x[test_instance])
            # print('Yess: ',[feature_info_box[k.split('-')[0]] for k in k2.annotate_placeholder.to_list()])
            # Get the info box for the tool tips
            attrs = att = k2.Values.to_numpy()
            attribution_dict = {fp: convVal(v) for fp, v in zip(taskObject.task_features.getFeaturesPlaceHolderNames(),
                                                                attrs)}
            feature_info_box = taskObject.task_features.getFeatureInfobox(values=feature_dict_vals, attributions=attribution_dict,
                                                                          scope_id=f'tx_{test_instance+1}',
                                                                          )
            infoBoxes = [feature_info_box[k.split(
                '-')[0]] for k in k2.annotate_placeholder.to_list()]
            ann = dx = k2  # FCC.xframe[prediction]
            placeholder_maps, [con, sup, ign] = FCC.getPlaceholderBasedSplit(k2, local_contradicting,
                                                                             local_supportive,
                                                                             local_ignore)
            feature_placeholders = json.dumps(placeholder_maps)
            # print(local_rank_list)
            # print(k2.annotate_placeholder_code.to_list())
            feature_division = json.dumps(
                {'rank': local_rank_list,
                 'annotate_code': k2.annotate_placeholder_code.to_list(),
                 'explainable_df': k2.to_json(),
                 'feature_type': k2.ftype.to_list(),
                 'contradict': con,
                 'support': sup,
                 'ignore': ign})

            # att, fig1, nb = local_exp.analyseInstanceExplainer(instance, prediction)
            # fig1.get_axes()[0].set_xlabel("Coefficients (impact on model output) \n Green = Positive Impact, \nRed = Negative Impact")

            fig1 = explainer_barplot(
                attrs, feature_names=ann.annotate_placeholder.to_list(), max_display=MAX_DISPLAY, show=False)
            plt.tight_layout()

            encoded_fig = utils.processFigureForDisplay(fig1)
            local_pred_exp = pd.DataFrame()

            dd = pd.DataFrame(abs(att), columns=['effect_abs'])
            dd['raw_attr'] = att
            dd['Variable'] = ann.Variable
            dd['Values'] = dataset_x[test_instance]
            dd['annotate_placeholder'] = ann.annotate_placeholder
            dd['Sign'] = np.where(dd['raw_attr'] > 0, 'green', 'red')

            dx = k3 = dd.sort_values(by='effect_abs', ascending=False)
            edf = dx.loc[:, ['Variable', 'annotate_placeholder', 'Values']]
            edf.reset_index(inplace=True)
            edf.drop('index', axis=1)
            edf.index = edf.index+1
            edf.drop('index', axis=1, inplace=True)
            placeholders = edf.to_html(
                classes="table table-striped table-hover eval_tb", index=False)
            local_feats = ann.annotate_placeholder.to_list()
            plotly_attr = sorted(zip(local_feats, attrs,
                                     k2['Sign'].to_list()),
                                 key=lambda x: abs(x[1]), reverse=True)
            gfeats, gattr, gcolors = [], [], []
            feat_pmap = {f'F{i+1}': v for i,
                         v in enumerate(dataset_x[test_instance])}
            tooltips = []
            for fn, attr, sign in plotly_attr:
                ll = fn.split('-')
                plc_holder = ll[0]
                feat_obj = taskObject.task_features.feature_set[plc_holder]
                if len(ll) == 1:
                    val_ = feat_pmap[fn]
                else:
                    val_ = int(ll[-1].replace('V', '').strip())
                    val_ = feat_obj.levels[val_]
                tooltips.append(feat_obj.getMiniInfo(val_, attr))

                gfeats.append(fn)
                gattr.append(attr)
                gcolors.append(sign)

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            confidence = 0
        max_display = MAX_DISPLAY

        # Get the actual class name for the prediction made

        cl_name = list(taskObject.classes_placeholders.keys())[
            prediction] if task_type != 'reg' else f'''{prediction:.2f}'''
        expected_pred = list(taskObject.classes_placeholders.keys())[
            int(correct_output)] if task_type != 'reg' else f'''{correct_output:.2f}'''

        top, moderate, low = utils.divideFeatures(gfeats[:])
        # print(moderate)
        top = combineList(top) if len(top) > 0 else ''
        moderate = combineList(moderate) if len(moderate) > 0 else ''
        low = combineList(low) if len(low) > 0 else ''

        mapped_question_placeholders = {'<pred>': cl_name,
                                        '<expected_pred>': expected_pred,
                                        '<low_features>': low,
                                        '<moderate_features>': moderate,
                                        '<top_features>': top, }
        # print(mapped_question_placeholders)
        annotator_questions = local_questions(
            mapped_placeholders=mapped_question_placeholders)

        # print(top, moderate, low)
        pred_likelihood = '<u>Prediction Likelihood</u> <br/>' + \
            ', '.join(class_probabilities) if task_type != 'reg' else ' '

        pred_x = f'''
         {cl_name} 
        ''' if task_type != 'reg' else f'''{cl_name}'''  # ({taskObject.classes_placeholders[cl_name].strip()})

        expected_pred_x = f'''
        {expected_pred}
        ''' if task_type != 'reg' else f'''{expected_pred}'''  # ({taskObject.classes_placeholders[expected_pred].strip()})

        pred_statement = f'''
        <div class="alert alert-success" role="alert">
                                                <p class="card-text">
                                                    
                                                    Model Prediction:  
                                <b>{pred_x}</b>
                               <!--  <br/> Expected Prediction:   <b>{expected_pred_x}</b> --> </p>
                                
                                <p> {pred_likelihood} 
                                                </p>
                                            </div>
        '''
        # test_pred.model_name = model_name
        # test_pred.task_name = task_name
        task_object = taskObject
        # print(task_object.task_summary)
        question_preamble = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                                        fill="currentColor" class="bi bi-info-circle"
                                                        viewBox="0 0 16 16">
                                                        <path
                                                            d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
                                                        <path
                                                            d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z" />
                                                    </svg> 
                                                    The graph shown illustrates how the features used to train 
                                                    a model contribute to the prediction  of the class label {cl_name} for the test example under consideration. 
                                                    <p>Please provide an analytical narrative summarizing the contributions
                                                    of the different features. </p>
                                                    The content of your narrative should answer the following: 
        '''
        evalQuestion = generateInputBoxesForQuestions(annotator_questions, box_info='narrations_questions_textbox') + \
            f''' <input type="hidden" id="local_questions" value="{annotator_questions}"/>'''
        # print(evalQuestion)
        evalQuestion = question_preamble+evalQuestion + \
            f''' <input type="hidden" id="label_pred_questions" value="{annotator_questions}"/>'''
        task_summary = annotationInstance.tasks.getTaskSummary(test_pred.task_name)
        print(test_pred.task_name,task_name)
        # question_preamble+annotator_questions
        return {
            'task_dict': task_object.asDictInfo() if task_object else {},
            # 'task_object': task_object if task_object else None,
            'task_summary': task_summary,  # task_object.task_summary,
            'model_name': test_pred.model_name,
            'task_name': test_pred.task_name,
            'task_feats_html': task_object.task_features.getFeatureInfoAsHtml(values={}, scope_id=f'desc_page', max_display_per_row=7) if task_object else '<div></div>',
            'plot': encoded_fig,
            'statement': pred_statement.strip(),
            'confidence': class_probabilities,#f'{confidence*100:.2f}%'
            'prediction': f'{cl_name}',
            'feat_div': feature_division,
            'feat_placeholder': feature_placeholders,
            'test_instance': test_instance,

            'placeholders': placeholders,
            'attr': gattr[:max_display][::-1],
            'feats': gfeats[:max_display][::-1],
            'color': gcolors[:max_display][::-1],
            'info_box': tooltips[:max_display][::-1],
            'annotator_questions': evalQuestion + f''' <input type="hidden" id="instance_pred_questions" value="{annotator_questions}"/>''',
            'features_display': feature_dict_vals_html}

min_annotation_count = 1


def pickMLTask(task_list, db, problem_type=''):
    annotationInstance = AnnotationSession()
    # task_list = list(annotationInstance.tasks.tasks_info.keys())
    random.shuffle(task_list)
    task_list = sorted(list(task_list))
    task_name = random.choice(task_list[::-1])

    counts = checkNumberOfAnnotations(db, task_name, problem_type)
    print(f'Task Name: {task_name} Counts: {counts}')
    alreadySubmitted = counts > min_annotation_count
    checked = 0
    while alreadySubmitted:
        if checked > 6:
            break
        #random.shuffle(task_list)
        task_list = sorted(list(task_list))
        task_name = random.choice(task_list[::-1])
        counts = checkNumberOfAnnotations(db, task_name, problem_type)
        alreadySubmitted = counts > min_annotation_count
        print(f'Task Name: {task_name} Counts: {counts}')
        if not alreadySubmitted:
            break
        else:
            checked += 1

    return task_name


@app.post("/compare_local_global")
async def getLocalGlobalComparison(test_pred: schema.PredictionRequest):
    # annotationInstance = AnnotationSession()
    annotationInstance = AnnotationSession()
    if test_pred.is_randomize == 1:
        task_name = pickMLTask(
            list(annotationInstance.tasks.tasks_info.keys()))
        # random.shuffle(task_list)
        # task_name = random.choice(task_list[::-1])
        # task_name = random.choice(task_list)

        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        test_pred.model_name = model_name
        test_pred.task_name = task_name
        dataset_x, data_y = taskObject.dataset['test_pack']
        test_pred.test_instance = np.random.choice(np.arange(len(dataset_x)))

    else:
        isValid, taskObject = annotationInstance.isValidTask(
            test_pred.task_name)
    # isValid, taskObject = annotationInstance.isValidTask(test_pred.task_name)
    task_type = taskObject.task_type
    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}
    actual_model = taskObject.modelIdx[test_pred.model_name]
    test_pred.model_name = actual_model
    model_details = taskObject[actual_model]
    current_model = model_details['model']
    task_type = taskObject.task_type
    if isinstance(current_model, NeuralModelInterface):
        current_model.loadModel()
    if current_model:
        # For now we will read in the global explanation for each classification
        FCC = taskObject.modelsGlobalExplainer[test_pred.model_name]
        test_instance = int(test_pred.test_instance)
        # Get the data instance
        dataset_x, data_y = taskObject.dataset['test_pack']

        x_test = pd.DataFrame([dataset_x[test_instance]],
                              columns=taskObject.task_features.getFeaturesNames())
        correct_output = data_y[test_instance]
        feat_pmap = {f'F{i+1}': v for i,
                     v in enumerate(dataset_x[test_instance])}

        prediction = taskObject.task_features.model_predict(current_model, x_test)[
            0]
        confidence = 0

        feature_dict_vals = {fp: utils.convVal(v) for fp, v in zip(taskObject.task_features.getFeaturesPlaceHolderNames(),
                             dataset_x[test_instance])}

        feature_dict_vals_html = taskObject.task_features.getFeatureInfoAsHtml(
            values=feature_dict_vals, scope_id=f'tp_{test_instance+1}', max_display_per_row=4)

        if task_type == 'cls':
            prediction = int(prediction)
            class_probs = taskObject.task_features.model_predict_proba(
                current_model, x_test)[0]
            confidence = class_probs[prediction]
            class_probabilities = [f'{c}: {p*100:.2f}%' for c, p in zip(
                list(taskObject.classes_placeholders.keys()), class_probs)]

        try:
            ann = dx = FCC.xframe[prediction] if task_type == 'cls' else FCC.xframe[0]

            feature_names = taskObject.task_features.getFeaturesNames()
            nb_features = len(taskObject.task_features.getFeaturesNames())

            # Get the local explanation using Lime
            local_exp = taskObject.modelsLocalExplainer[test_pred.model_name]
            if isinstance(local_exp, LRPExplainerInterface):
                local_exp = LRPExplainerInterface(
                    taskObject.task_features, current_model.model)
                instance, pred = local_exp.explain_data_row(
                    [dataset_x[test_instance]],)
            else:
                instance = local_exp.explain_data_row([dataset_x[test_instance]],
                                                      model=current_model,
                                                      labels=list(
                                                      range(len(taskObject.task_classes))),
                                                      num_features=20)
            # instance = local_exp.explain_data_row([dataset_x[test_instance]], model=current_model,
            #                                      labels=list(range(len(taskObject.task_classes))), num_features=20)
            # processAttribution(instance.local_exp,nb_features,)[prediction]
            attrs = att = instance[prediction] if task_type == 'cls' else instance[1]
            k2, local_rank_list, [local_contradicting,
                                  local_supportive,
                                  local_ignore] = FCC.splitAttribution(attrs, MAX_DIV_FEATURES=MAX_DIV_FEATURES,
                                                                       feature_names=taskObject.task_features.getFeaturesNames())
            k2 = localExplainerPostProcess(
                taskObject.task_features, k2, dataset_x[test_instance], local_mode=False)
            # print(k2)

            ann = dx = k2  # FCC.xframe[prediction]

            # Get the features mapped based on the raw placeholder column
            placeholder_maps, [con, sup, ign] = FCC.getPlaceholderBasedSplit(k2, local_contradicting,
                                                                             local_supportive,
                                                                             local_ignore)

            # Get the features mapped based on the annotate_placeholder column
            placeholder_maps_ann, [con_ann, sup_ann, ign_ann] = FCC.getPlaceholderBasedSplit(k2, local_contradicting,
                                                                                             local_supportive,
                                                                                             local_ignore, use_simple=True)

            feature_placeholders = json.dumps(
                {"raw_placeholder": placeholder_maps, "annotate_placeholder": placeholder_maps_ann})
            # print(local_rank_list)
            # print(k2.annotate_placeholder_code.to_list())
            feature_division = json.dumps(
                {'rank': local_rank_list,
                 'annotate_code': k2.annotate_placeholder_code.to_list(),
                 'feature_type': k2.ftype.to_list(),
                 'explainable_df': k2.to_json(),
                 'contradict': {"raw_placeholder": con, "annotate_placeholder": con_ann},
                 'support': {"raw_placeholder": sup, "annotate_placeholder": sup_ann},
                 'ignore': {"raw_placeholder": ign,
                            "annotate_placeholder": ign_ann}})
            # print([con_ann, sup_ann, ign_ann])

            # att, fig1, nb = local_exp.analyseInstanceExplainer(instance, prediction)
            # fig1.get_axes()[0].set_xlabel("Coefficients (impact on model output) \n Green = Positive Impact, \nRed = Negative Impact")
            attrs = att = ann.Values.to_numpy()
            fig1 = explainer_barplot(
                attrs, feature_names=ann.annotate_placeholder.to_list(), max_display=MAX_DISPLAY, show=False)
            plt.tight_layout()

            # encoded_fig = processFigureForDisplay(fig1)
            placeholders = k2.to_html(
                classes="table table-striped table-hover eval_tb", index=False)
            local_feats = ann.annotate_placeholder.to_list()
            plotly_attr = sorted(zip(local_feats, attrs,
                                     k2['Sign'].to_list()),
                                 key=lambda x: abs(x[1]), reverse=True)
            lfeats, lattr, lcolors = [], [], []
            # print(plotly_attr)
            local_tooltips = []
            for fn, attr, sign in plotly_attr:
                ll = fn.split('-')
                plc_holder = ll[0]
                feat_obj = taskObject.task_features.feature_set[plc_holder]
                if len(ll) == 1:
                    val_ = feat_pmap[fn]
                else:
                    val_ = int(ll[-1].replace('V', '').strip())
                    val_ = feat_obj.levels[val_]
                attr = round(attr, 3)
                sign = sign if attr != 0 else 'yellow'
                local_tooltips.append(feat_obj.getMiniInfo(val_, attr))
                lfeats.append(fn)
                lattr.append(attr)
                lcolors.append(sign)

            ltop, lmoderate, llow = utils.divideFeatures(lfeats[:])
            ltop = combineList(ltop) if len(ltop) > 0 else ' '
            lmoderate = combineList(lmoderate) if len(lmoderate) > 0 else ' '
            llow = combineList(llow) if len(llow) > 0 else ' '

            cl_name = list(taskObject.classes_placeholders.keys())[
                prediction] if task_type != 'reg' else f'''{prediction:.3f}'''

            # Process the Global Features
            class_xframe = FCC.xframe[prediction] if task_type != 'reg' else FCC.xframe[0]
            # Do the post-processing to handle the categorical features
            class_xframe, [global_rank_list, [global_contradicting,
                                              global_supportive,
                                              global_ignore]] = globalExplainerPostProcess(taskObject.task_features, class_xframe)

            # Get the global features with respect to the local features
            accepted_global, global_rank_list, [global_contradicting,
                                                global_supportive,
                                                global_ignore] = localGlobalOverLap(k2, class_xframe)
            global_placeholder_maps, [gcon, gsup, gign] = FCC.getPlaceholderBasedSplit(accepted_global, global_contradicting,
                                                                                       global_supportive,
                                                                                       global_ignore)
            # Get the features mapped based on the annotate_placeholder column
            global_placeholder_maps_ann, [gcon_ann, gsup_ann, gign_ann] = FCC.getPlaceholderBasedSplit(accepted_global, global_contradicting,
                                                                                                       global_supportive,
                                                                                                       global_ignore, use_simple=True)
            # print('Spliots', [gcon, gsup, gign])

            # SPlit the contributions from both perspectives
            comparative_contribution_split = FCC.compareExplainables(source_perspective=[
                                                                     con_ann, sup_ann, ign_ann], reference_perspective=[gcon_ann, gsup_ann, gign_ann])
            print(f'Globals:   {[gcon_ann, gsup_ann, gign_ann]}')
            print(f'Locals:    {[con_ann, sup_ann, ign_ann]}')
            print(f'Separated: {comparative_contribution_split}')

            dx = accepted_global
            placeholders = accepted_global.to_html(
                classes="table table-striped table-bordered table-hover table-sm eval_tb", index=False)
            # fig = FCC.plotImportance(label=class_index)

            gk2 = accepted_global.sort_values(by='effect_abs', ascending=False)
            global_att = gk2.effect_raw.to_numpy()
            # print(gk2)
            global_feats = accepted_global.annotate_placeholder_code.to_list()
            global_feature_placeholders = json.dumps(global_placeholder_maps)
            global_feature_division = json.dumps({'rank': global_rank_list,
                                                  'annotate_code':  accepted_global.annotate_placeholder_code.to_list(),
                                                  'explainable_df': accepted_global.to_json(),
                                                  'feature_type': accepted_global.ftype.to_list(),
                                                  'contradict': gcon,
                                                  'support': gsup,
                                                  'ignore': gign})
            # print(k2.Sign.to_list())
            plotly_attr = sorted(zip(global_feats, accepted_global.effect_raw.to_list(), accepted_global.Sign.to_list()),
                                 key=lambda x: abs(x[1]), reverse=True)
            # print(plotly_attr)
            gfeats, gattr, gcolors = [], [], []
            global_tooltips = []
            for fn, attr, sign in plotly_attr:

                ll = fn.split('-')
                plc_holder = ll[0]
                feat_obj = taskObject.task_features.feature_set[plc_holder]

                val = None if len(ll) == 1 else int(
                    ll[-1].replace('V', '').strip())
                val_ = val
                if val:
                    val_ = feat_obj.levels[val]

                global_tooltips.append(feat_obj.getMiniInfo(val_, attr))

                attr = round(attr, 3)
                sign = sign if attr != 0 else 'yellow'
                gfeats.append(fn)
                gattr.append(attr)

                gcolors.append(sign)

            gtop, gmoderate, glow = utils.divideFeatures(gfeats[:])
            gtop = combineList(gtop) if len(gtop) > 0 else ' '
            gmoderate = combineList(gmoderate) if len(gmoderate) > 0 else ' '
            glow = combineList(glow) if len(glow) > 0 else ' '

            mapped_compare_preamble_question_placeholders = {'<pred>': cl_name,
                                                             '<low_local>': llow,
                                                             '<moderate_local>': lmoderate,
                                                             '<top_local>': ltop,
                                                             '<low_global>': glow,
                                                             '<moderate_global>': gmoderate,
                                                             '<top_global>': gtop,
                                                             }
            preample_annotator_questions = compare_preamble_questions(
                mapped_placeholders=mapped_compare_preamble_question_placeholders)

            # print(preample_annotator_questions)
            # comparative_contribution_split
            # Create a mapper for picking the questions set for the deep comparative narratives
            mapped_compare_question_placeholders = mapped_compare_preamble_question_placeholders
            mapped_compare_question_placeholders['<GS_features>'] = combineList(
                comparative_contribution_split['concensus_support'])
            mapped_compare_question_placeholders['<GS_features>'] = combineList(
                comparative_contribution_split['concensus_support'])
            mapped_compare_question_placeholders['<GC_features>'] = combineList(
                comparative_contribution_split['concensus_contradict'])
            mapped_compare_question_placeholders['<GSC_features>'] = combineList(comparative_contribution_split[
                'concensus_support'][:2]+comparative_contribution_split['concensus_contradict'][:2])
            mapped_compare_question_placeholders['<EBS_features>'] = combineList(
                comparative_contribution_split['evidence_based_support'])
            mapped_compare_question_placeholders['<EBC_features>'] = combineList(
                comparative_contribution_split['evidence_based_contradict'])
            mapped_compare_question_placeholders['<EBSC_features>'] = combineList(comparative_contribution_split[
                'evidence_based_support'][:3]+comparative_contribution_split['evidence_based_contradict'][:3])

            # Formulate the comparative questions.
            annotator_questions = preample_annotator_questions+compare_questions(
                mapped_placeholders=mapped_compare_question_placeholders)

            max_display = MAX_DISPLAY
            hidden_place_holders = f'''
            <input type="hidden" id="feat_div_global_compare-impact" value="" readonly />
            <input type="hidden" id="feat_placeholder_global_compare-impact" value="" readonly />
            <input type="hidden" id="feat_div_local_compare-impact" value="" readonly />
            <input type="hidden" id="feat_placeholder_local_compare-impact" value="" readonly />
            '''.strip()

            # Info Boxes
            feature_info_box = taskObject.task_features.getFeatureInfobox(values=feature_dict_vals,
                                                                          scope_id=f'tx_{test_instance+1}',
                                                                          )
            local_infoBoxes = [feature_info_box[k.split(
                '-')[0]] for k in k2.annotate_placeholder.to_list()]
            global_infoBoxes = [feature_info_box[k.split(
                '-')[0]] for k in accepted_global.annotate_placeholder.to_list()]

            # Call the explainer Interface here
            cl_name = list(taskObject.classes_placeholders.keys())[
                prediction] if task_type != 'reg' else f'''{prediction:.3f}'''
            expected_pred = list(taskObject.classes_placeholders.keys())[
                int(correct_output)] if task_type != 'reg' else f'''{correct_output:.3f}'''

            pred_likelihood = '<u>Prediction Likelihood</u> <br/>' + \
                ', '.join(class_probabilities) if task_type != 'reg' else ' '
            pred_x = f'''{cl_name} ({taskObject.classes_placeholders[cl_name].strip()})''' if task_type != 'reg' else f'''{cl_name}'''
            expected_pred_x = f'''
            {expected_pred} ({taskObject.classes_placeholders[expected_pred].strip()}) 
            ''' if task_type != 'reg' else f'''{expected_pred}'''

            pred_statement = f'''
        <div class="alert alert-success" role="alert">
                                                <p class="card-text">
                                                    
                                                    Model Prediction:  
                                <b>{pred_x}</b> 
                                <!--  <br/> Expected Prediction:   <b>{expected_pred_x}</b> -->
                                
                                <br/> {pred_likelihood} 
                                                </p>
                                            </div>
        '''
            max_display = MAX_DISPLAY
            # 'plot': encoded_fig,
            question_preamble = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                                        fill="currentColor" class="bi bi-info-circle"
                                                        viewBox="0 0 16 16">
                                                        <path
                                                            d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
                                                        <path
                                                            d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z" />
                                                    </svg> 
                                                    The graph shown illustrates how the features used to train 
                                                    a model contribute to the prediction  of the class label {cl_name} for the test example under consideration. 
                                                    <p>Please provide an analytical narrative summarizing the contributions
                                                    of the different features. </p>
                                                    The content of your narrative should answer the following: 
        '''
            evalQuestion = generateInputBoxesForQuestions(annotator_questions, box_info='narrations_questions_textbox') + \
                f''' <input type="hidden" id="local_questions" value="{annotator_questions}"/>'''
            evalQuestion = question_preamble+evalQuestion + \
                f''' <input type="hidden" id="label_pred_questions" value="{annotator_questions}"/>'''
            task_summary = annotationInstance.tasks.getTaskSummary(task_name)
            return {'plot': '',
                    'task_summary': task_summary,
                    'model_name': test_pred.model_name,
                    'task_name': test_pred.task_name,
                    'statement': pred_statement.strip(),
                    'confidence': f'{confidence*100:.2f}%' if task_type != 'reg' else 'N/A',
                    'prediction': f'{cl_name}',
                    'feat_div': feature_division,
                    'feat_placeholder': feature_placeholders,
                    'global_feat_div': global_feature_division,
                    'global_feat_placeholder': global_feature_placeholders,
                    'test_instance': test_instance,
                    'placeholders': placeholders,
                    'local_attr': lattr[:max_display][::-1],
                    'local_feats': lfeats[:max_display][::-1],
                    'local_color': lcolors[:max_display][::-1],
                    'features_display': feature_dict_vals_html,
                    'global_attr': gattr[:max_display][::-1],
                    'global_feats': gfeats[:max_display][::-1],
                    'global_color': gcolors[:max_display][::-1],
                    'compare_hidden_placeholder': hidden_place_holders,
                    'local_info_box': local_tooltips[:max_display][::-1],
                    'global_info_box': global_tooltips[:max_display][::-1],
                    'annotator_questions': evalQuestion+f''' <input type="hidden" id="instance_compare_pred_questions" value="{annotator_questions}"/>'''
                    }
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            confidence = 0


@app.post("/get_model_performance")
async def getModelPerformance(model_req: schema.TaskModel):
    randomly_generate = np.random.choice([1, 2, 2, 1])
    # if randomly_generate == 1:
    #    return generateRandomPerformanceMetrics()
    annotationInstance = AnnotationSession()
    # isValid, taskObject = annotationInstance.isValidTask(model_req.task_name)

    # Randomly pick a task
    task_name = pickMLTask(list(annotationInstance.tasks.tasks_info.keys()))
    # task_name ='Job Change of Data Scientists'
    # task_list = list(annotationInstance.tasks.tasks_info.keys())
    # random.shuffle(task_list)
    # task_list =sorted(list(task_list))
    # task_name = random.choice(task_list[::-1])
    # task_name='Airline Passenger Satisfaction'

    isValid, taskObject = annotationInstance.isValidTask(task_name)

    tasks_models = list(taskObject.modelIdx.keys())
    nb_models = utils.validateURLCode(
        model_req.nb_models.upper(), acceptable_extension_codes)
    # print(nb_models)
    if nb_models is None:
        resp = RedirectResponse(
            url='/PerformanceNarratives/error', status_code=status.HTTP_404_NOT_FOUND)
        return resp
    # nb_models = np.random.choice([1, 2, 3, 4, 4, len(tasks_models)-1])
    modelss = sorted(list(taskObject.modelIdx.keys()))
    model_names = random.sample(modelss, k=nb_models)
    model_req.model_name = model_names
    # ({{task_object.classes_placeholders[cl_name] |capitalize}})
    task_classes = list(taskObject.classes_placeholders.keys())
    # [f'{cl_name}({taskObject.classes_placeholders[cl_name]})' for cl_name in task_classes]
    classes = '<p> Class Labels: <b>' + \
        ', '.join(task_classes[:-1])+'</b> and <b>'+task_classes[-1]+'</b></p>'

    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}

    # if valid model

    task_dataset = taskObject.dataset
    # print(task_dataset)
    task_summary = annotationInstance.tasks.getTaskSummary(task_name)

    # print(task_dataset)
    dataset_info = classes+task_dataset.get('dataset_class_distribution', '')
    if len(model_names) > 0:
        # Credit_Card_Fraud
        evalMetrics = [getModelMetrics(taskObject, model_name)
                       for model_name in model_names]
        # evalMetrics = [dict(zip(evalMetrics[0], random.sample(list(evalMetrics[0].values()), len(evalMetrics[0]))))]
        print(evalMetrics)
        edf = pd.DataFrame([list(e.values()) for e in evalMetrics])

        metriclist = list(evalMetrics[0].keys())
        metrics_definitions = utils.getMetricInformation(metriclist)
        edf.columns = metriclist

        mids = ['A', 'B', 'C', 'D', 'E']
        models_idx = [f'Model {mids[i]}' for i in range(nb_models)]
        edf.index = models_idx
        edf = edf.round(2)
        temp_cols = edf.columns.to_list()
        random.shuffle(temp_cols)
        edf = edf[temp_cols]
        metric_info = f'''<input id="eval_metric_info" type="hidden" value="{json.dumps(edf.to_json())}"/> 
         <input id="task_name_metric" type="hidden" value="{task_name }"/> 
         <input id="slt_model_perform_metric" type="hidden" value="{"<#>".join(model_names)}"/> 
        '''
        met_assessment = composeMetricEvalTable(
            edf.columns.to_list(), nb_models)
        evalQuestion = met_assessment+composeQuestions(
            edf, nb_models, evalMetrics)+metric_info

        # print(evalQuestion, metric_mapped_placeholders['<rand_metrics>'])

        return {'metric_table': edf.to_html(classes="table table-striped table-hover eval_tb"),
                'metric_values': json.dumps(edf.to_json()),
                'metrics_definitions': metrics_definitions,
                'task_name': task_name,
                'task_summary': task_summary,  # taskObject.task_summary,
                'metric_info': '',
                'evalQuestion': evalQuestion,
                'dataset_info': dataset_info}


def composeMetricEvalTable(metrics, nb_models):
    if nb_models > 1:
        return ''
    else:
        # Compose a table with the metrics along with radio buttons to
        preamble = '''
        <div class="alert alert-info annotation_info_box" role="alert"><p><svg width="24" height="24" xmlns="http://www.w3.org/2000/svg" 
    fill-rule="evenodd" clip-rule="evenodd"><path d="M12 0c6.623 0 12 5.377 12 12s-5.377 12-12 12-12-5.377-12-12 5.377-12 12-12zm0 1c6.071 0 11 4.929 11 11s-4.929 11-11 11-11-4.929-11-11 4.929-11 11-11zm.5 17h-1v-9h1v9zm-.5-12c.466 0 .845.378.845.845 0 .466-.379.844-.845.844-.466 0-.845-.378-.845-.844 0-.467.379-.845.845-.845z"/></svg>
Rate the score achieved for each evaluation metric.
</p>
                        </div>
        <table border="1" class="table table-striped table-hover tbl_metric_assessment"><thead><th>Metric</th><th>Assessment</th></thead><tbody>'''

        inner = ''
        for idx, m in enumerate(metrics):
            inner += f'''<tr><td>{m}</td><td class="metric_assessment">
            
            <span class="table_rb_items"> <label>Very Low</label><input type="radio" id="{m}" name="m{idx+1}_rb" value="1" style="width: 1.25rem;
    height: 1.25rem;"/> </span> |
            <span class="table_rb_items"> <label>Low</label><input type="radio"  id="{m}" name="m{idx+1}_rb" value="2" style="width: 1.25rem;
    height: 1.25rem;"/> </span> |
            <span class="table_rb_items"> <label>Moderate</label><input type="radio"  id="{m}" name="m{idx+1}_rb" value="3" style="width: 1.25rem;
    height: 1.25rem;"/> </span> |
            <span class="table_rb_items"> <label>High</label><input type="radio"  id="{m}" name="m{idx+1}_rb" value="4" style="width: 1.25rem;
    height: 1.25rem;"/> </span> |
            <span class="table_rb_items"> <label>Very High</label><input type="radio"  id="{m}" name="m{idx+1}_rb" value="5" style="width: 1.25rem;
    height: 1.25rem;"/> </span> |
</td></tr>'''
        terminate = '''<tbody></table> '''  # <button id="rd_btn" onclick="getOptions();">Compile</button>

        return preamble+inner+terminate


@app.post('/validate_eval_narrative')
async def validatePerformanceNarrative(request: Request,
                                       narr: schema.ValidateSubmittedNarrative,
                                       db: Session = Depends(get_db)):
    conclusion = 'Submitting narrative now.'
    status = 0
    return {'status': status, 'message': conclusion}


@app.post('/validate_eval_narrative_bk')
async def validatePerformanceNarrative(request: Request, narr: schema.ValidateSubmittedNarrative, db: Session = Depends(get_db)):

    current_user = checkisLogin(request, force_template=False)
    if (isinstance(current_user, dict) and current_user.get('id', None)):
        narratives = getAllPerformanceNarrationsForUser(
            db, user_id=int(current_user['id']), return_last=20)
        similarity = utils.checkNarrativesSimilarities(narr.narration, narratives)
        # print(similarity)
        conclusion = ''
        if similarity > 0.75:
            conclusion = 'The narrative provided is somewhat identical to some of your previous submissions. Please edit and try submitting again. '
            status = -1
        else:
            conclusion = 'Submitting narrative now.'
            status = 0
        return {'status': status, 'message': conclusion}


def getAnnotationRewardCode(task_type, db, annotation_id, request: Request):
    nb_parts = 3
    part_len = 5
    ann_info = getAnnotatorInfo(request)
    if task_type != 'performance':
        nb_parts = 4
        part_len = 4

    all_past_codes = set(getAllSubmitedCodes(db))
    ann_c = random.choice(['_', '-'])
    redeemCode = cc_generate(n_parts=nb_parts, part_len=part_len) + \
        ann_c+str(annotation_id)+'-'+task_mapper[task_type]
    while(True):
        if redeemCode in all_past_codes:
            redeemCode = cc_generate(
                n_parts=nb_parts, part_len=part_len)+ann_c+str(annotation_id)+'-'+task_mapper[task_type]
        else:
            break
    idx = [annotation_id]
    if task_type == 'performance':
        up = addPerformanceAnnotationRedeemCode(db, idx, redeemCode, 45)
    elif task_type == 'local':
        up = addLocalAnnotationRedeemCode(db, idx, redeemCode, 45)
    elif task_type == 'global':
        up = addGlobalAnnotationRedeemCode(db, idx, redeemCode, 45)
    elif task_type == 'compare':
        up = addCompareAnnotationRedeemCode(db, idx, redeemCode, 45)

    return redeemCode  # {'code':redeemCode}


@app.post('/eval_narrative')
async def add_eval_narrative(request: Request, eval_narrative: schema.EvalMetricNarrative, db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    # Get the annotator IP address
    annotator_info = getAnnotatorInfo(request)

    # print(eval_narrative.metrics_values)
    if eval_narrative.task_name != 'randomized':
        # 'Model-3 <#> Model-4 <#> Model-1'
        model_names = [m.strip()
                       for m in eval_narrative.model_name.split('<#>')]
        nb_models = len(model_names)
        isValid, taskObject = annotationInstance.isValidTask(
            eval_narrative.task_name)
        actual_model = ''  # taskObject.modelIdx[eval_narrative.model_name]
        # eval_narrative.model_name = eval_narrative.model_name
        # model_details = taskObject[actual_model]
        if model_names:
            evalMetrics = [getModelMetrics(
                taskObject, model_name) for model_name in model_names]
            edf = pd.DataFrame([list(e.values()) for e in evalMetrics])
            metriclist = list(evalMetrics[0].keys())
            metrics_definitions = utils.getMetricInformation(metriclist)
            edf.columns = metriclist
            mids = ['A', 'B', 'C', 'D', 'E']
            models_idx = [f'Model {mids[i]}' for i in range(nb_models)]
            edf.index = models_idx
            edf = edf.round(2)
            temp_cols = edf.columns.to_list()
            random.shuffle(temp_cols)
            edf = edf[temp_cols]
            # confusionMatrix = model_details['eval_metrics'].get(
            #    'confusion_matrix', None)
            # evalMetrics = {k: v for k, v in model_details['eval_metrics'].items() if k not in [
            #    'confusion_matrix', 'cm']}
            # edf = pd.DataFrame([list(evalMetrics.values())])
            # metriclist = list(evalMetrics.keys())
            # edf.columns = metriclist

            metric_info = json.dumps(edf.to_json())
            is_datasetimBalance = taskObject.dataset.get('is_imbalance', None)
            if is_datasetimBalance is not None:
                is_datasetimBalance = 1 if is_datasetimBalance else 2
            else:
                is_datasetimBalance = 2
            isValid = True

    elif eval_narrative.task_name == 'randomized':
        model_details = {'model': 'randomized',
                         }
        # model_details['eval_metrics'] =
        isValid = True
        metric_info = eval_narrative.metrics_values
    else:
        isValid = False
    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected', 'tn': eval_narrative.task_name}

    # current_user = checkisLogin(request, force_template=False)
    if isValid:
        # perform checks to make sure they are not just submitting the same narratives over and over
        try:
            db_data = db_tables.EvaluationMetricNarration()
            db_data.metrics_values = metric_info  # json.dumps(edf.to_json())
            db_data.task_name = eval_narrative.task_name
            db_data.narration = eval_narrative.narration
            db_data.narrator = 45  # eval_narrative.narrator
            db_data.model_name = eval_narrative.model_name
            db_data.narrative_question = eval_narrative.narrative_question
            db_data.dataset_info = eval_narrative.dataset_info
            db_data.nb_models = nb_models
            annotator_info = getAnnotatorInfo(request)
            db_data.user_ip = annotator_info['ip']
            db_data.is_dataset_balanced = is_datasetimBalance
            db_data.imetric_score_rate = eval_narrative.metric_score_rate
            # print(eval_narrative.metric_score_rate)
            new_data = applyDbCommit(db, db_data)

            code = getAnnotationRewardCode(
                'performance', db, new_data.id, request)
            # db.add(db_data)
            # db.commit()
            return {'status': 'Success',
                    'Message': 'Narrative Saved',
                    'saved_data': new_data,
                    'redeem_code': code}
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return {'status': 'Error', 'Message': 'Unable to add narrative'}
    else:
        return {'status': 'Error', 'Message': 'Invalid Model'}


# Get the global variable importance
@app.post('/get_label_global_importance')
async def get_label_global_importance(label_global_request: schema.GlobalImportance, db: Session = Depends(get_db)):

    test_pred = model_req = label_global_request
    annotationInstance = AnnotationSession()
    test_pred = model_req
    selected_task = None
    if test_pred.task_name.strip() == 'randomize':
        # task_list = list(annotationInstance.tasks.tasks_info.keys())
        # random.shuffle(task_list)
        # task_name = random.choice(task_list[::-1])
        #task_name = pickMLTask(list(annotationInstance.tasks.tasks_info.keys()))
        task_name = pickMLTask( list(annotationInstance.tasks.tasks_info.keys()), db, problem_type='global')

        isValid, taskObject = annotationInstance.isValidTask(task_name)
        model_name = random.choice(list(taskObject.modelIdx.keys()))
        model_req.model_name = model_name
        model_req.task_name = task_name
        task_classes = list(taskObject.classes_placeholders.keys())
        label_global_request.class_label = random.choice(task_classes)
        selected_task = task_name

    else:
        isValid, taskObject = annotationInstance.isValidTask(
            model_req.task_name)
        selected_task = model_req.task_name
    isValid, taskObject = annotationInstance.isValidTask(model_req.task_name)
    if not isValid:
        # print('Hello')
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}

    actual_model = taskObject.modelIdx[model_req.model_name]
    model_req.model_name = actual_model
    model_details = taskObject[actual_model]
    task_type = taskObject.task_type

    # print(actual_model)
    if model_details['model']:
        FCC = taskObject.modelsGlobalExplainer[model_req.model_name]

        # If regression problem, we will want to choose the single dataframe saved by the global Explainer
        class_index = list(taskObject.classes_placeholders.keys()).index(
            label_global_request.class_label) if task_type == 'cls' else 0
        cl_name = label_global_request.class_label if task_type == 'cls' else '<b>HIGH-TARGET </b> value'
        # print(cl_name+'  KKOP ', class_index)
        class_xframe = FCC.xframe[class_index]
        # Do the post-processing to handle the categorical features
        # print(FCC.xframe)
        class_xframe, [global_rank_list, [global_contradicting,
                                          global_supportive,
                                          global_ignore]] = Task.globalExplainerPostProcess(taskObject.task_features, class_xframe)

        dx = class_xframe
        placeholders = class_xframe.to_html(
            classes="table table-striped table-bordered table-hover table-sm eval_tb", index=False)
        # fig = FCC.plotImportance(label=class_index)

        k2 = class_xframe.sort_values(
            by='effect_abs', ascending=False)

        global_att = k2.effect_raw.to_numpy()
        # global_feats = FCC.xframe[class_index].Variable.to_list()
        global_feats = k2.annotate_placeholder_code.to_list()

        fig = explainer_barplot(
            global_att, feature_names=global_feats, max_display=MAX_DISPLAY, show=False)
        plt.tight_layout()
        encoded_fig = Task.processFigureForDisplay(fig)

        # Split the attribution as either contrastive, supportive, ignore
        # Then use them to build the feature-placeholder and division
        attrs = global_att

        placeholder_maps, [con, sup, ign] = FCC.getPlaceholderBasedSplit(class_xframe, global_contradicting,
                                                                         global_supportive,
                                                                         global_ignore)
        # print('Globals: ')
        # print(class_xframe.annotate_placeholder_code.to_list())
        feature_placeholders = json.dumps(placeholder_maps)
        feature_division = json.dumps({'rank': global_rank_list,
                                       'annotate_code': class_xframe.annotate_placeholder_code.to_list(),
                                       'explainable_df': class_xframe.to_json(),
                                       'contradict': con,
                                       'support': sup,
                                       'ignore': ign})
        # print(k2.Sign.to_list())
        plotly_attr = sorted(zip(global_feats, class_xframe.effect_raw.to_list(), k2.Sign.to_list()),
                             key=lambda x: abs(x[1]), reverse=True)
        # print(plotly_attr)
        gfeats, gattr, gcolors = [], [], []
        info_box = []
        pos_feats = []
        neg_feats = []
        pos_norm_feats = []
        neg_norm_feats = []
        pos_cat_feats = []
        neg_cat_feats = []
        limited = []
        limited_cat = []
        with_categorical = False
        for fn, attr, sign in plotly_attr:
            # global_infoBoxes = [feature_info_box[k.split('-')[0]] for k in accepted_global.annotate_placeholder.to_list()]
            # Get the info boxes for the tooltips
            attr = round(attr, 3)
            sign = sign if attr != 0 else 'yellow'
            ll = fn.split('-')
            plc_holder = ll[0]
            feat_obj = taskObject.task_features.feature_set[plc_holder]

            if len(ll) == 1:
                val_ = None
            else:
                val_ = int(ll[-1].replace('V', '').strip())
                # print(fn, val_, feat_obj.levels)
                val_ = feat_obj.levels[val_]

            # val = None if len(ll) == 1 else int(
            #     ll[-1].replace('V', '').strip())
            # val_ = val
            # if val:
            #    val_ = feat_obj.levels[val]

            info_box.append(feat_obj.getMiniInfo(val_, attr))
            gfeats.append(fn)
            gattr.append(attr)
            gcolors.append(sign)

            if sign == 'red':
                neg_feats.append(fn)
                if len(ll) > 1:
                    neg_cat_feats.append(plc_holder)
                else:
                    neg_norm_feats.append(fn)
            elif sign == 'green':
                if len(ll) > 1:
                    pos_cat_feats.append(plc_holder)
                else:
                    pos_norm_feats.append(fn)
                pos_feats.append(fn)

            else:
                limited.append(fn)
                if len(ll) > 1:
                    limited_cat.append(plc_holder)
        max_display = MAX_DISPLAY

        top, moderate, low = utils.divideFeatures(gfeats[:])
        top = combineList(top)
        moderate = combineList(moderate)
        low = combineList(low)

        norminal_features = list(set(pos_norm_feats[:4]+neg_norm_feats[:4]))
        categorical_features = list(set(pos_cat_feats[:4]+neg_cat_feats[:4]))
        mixed_pn = random.sample(norminal_features, 3) if len(
            norminal_features) > 2 else norminal_features
        mixed_pn_cat = random.sample(categorical_features, 3) if len(
            categorical_features) > 2 else categorical_features

        impact_mapped_placeholders = {'<mix_pos_neg_features>': combineList(mixed_pn) if len(mixed_pn) > 0 else ' ',
                                      '<mix_pos_neg_norm_features>': combineList(mixed_pn) if len(mixed_pn) > 0 else ' ',
                                      '<mix_pos_neg_cat_features>': combineList(mixed_pn_cat) if len(mixed_pn_cat) > 0 else ' ',
                                      '<neg_features>': combineList(neg_feats[:4]) if len(neg_feats) > 0 else ' ',
                                      '<pos_features>': combineList(pos_feats[:4]) if len(pos_feats) > 0 else ' ',
                                      '<low_features>': low,
                                      '<moderate_features>': moderate,
                                      '<top_features>': top,
                                      '<pred>': cl_name,

                                      }
        # <input type="hidden" id="feat_placeholder_global_impact" value='{feature_placeholders}' readonly />
        question_preamble = f'''
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
                                                        fill="currentColor" class="bi bi-info-circle"
                                                        viewBox="0 0 16 16">
                                                        <path
                                                            d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z" />
                                                        <path
                                                            d="m8.93 6.588-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 6.588zM9 4.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0z" />
                                                    </svg> 
                                                    The graph shown illustrates how different  features used to train 
                                                    a model generally affect it's classification performance across multiple test samples
                                                     in terms of predicting the 
                                                    class label {cl_name}.  <div class="dropdown-divider"></div><hr class="mt-2 mb-3" />
                                                    <p>Please provide an analytical narrative summarizing the contributions
                                                    of the different features based on the following questions.</p>
        '''
        label_question = global_questions(
            mapped_placeholders=impact_mapped_placeholders) if len(mixed_pn_cat) < 1 else global_questions_categorical(mapped_placeholders=impact_mapped_placeholders)
        # print(label_question)
        evalQuestion = generateInputBoxesForQuestions(label_question, box_info='narrations_questions_textbox') + \
            f''' <input type="hidden" id="eval_metric_questions" value="{label_question}"/>'''
        # print(evalQuestion)
        evalQuestion = question_preamble+evalQuestion + \
            f''' <input type="hidden" id="label_pred_questions" value="{label_question}"/>'''
        # print(label_question,)

        hidden_place_holders = f'''
        <input type="hidden" id="feat_div_global_impact" value="" readonly />
        <input type="hidden" id="feat_placeholder_global_impact" value="" readonly />
        <input id="slt_model_global" type="hidden" value="{model_name}"/>
        '''.strip()
        # print(hidden_place_holders)
        return {'plot': encoded_fig,
                'model_name': model_req.model_name,
                'task_name': selected_task,
                'cls_label': label_global_request.class_label,
                'placeholders': placeholders,
                'attr': gattr[:max_display][::-1],
                'feats': gfeats[:max_display][::-1],
                'color': gcolors[:max_display][::-1],
                'feat_div': feature_division,
                'feat_placeholder': feature_placeholders,
                'hidden_place_holder': hidden_place_holders,
                'global_info_boxes': info_box[:max_display][::-1],
                'label_questions': evalQuestion,  # label_question,
                'task_dict': taskObject.asDictInfo() if taskObject else {},
                # 'task_object': task_object if task_object else None,
                'task_summary': taskObject.task_summary,
                'task_feats_html': taskObject.task_features.getFeatureInfoAsHtml(values={}, scope_id=f'desc_page', max_display_per_row=7) if taskObject else '<div></div>',
                }


@app.post('/global_narrative')
async def add_global_narrative(request: Request, global_narrative: schema.GlobalNarrative, db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(
        global_narrative.task_name)
    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected', 'task_name': global_narrative.task_name}

    # actual_model = taskObject.modelIdx[global_narrative.model_name]
    try:
        actual_model = taskObject.modelIdx[global_narrative.model_name]
    except:
        actual_model = global_narrative.model_name
    global_narrative.model_name = actual_model
    model_details = taskObject[actual_model]
    model_details = taskObject[global_narrative.model_name]
    # current_user = checkisLogin(request, force_template=False)
    task_type = taskObject.task_type
    if model_details['model']:
        FCC = taskObject.modelsGlobalExplainer[global_narrative.model_name]
        # class_index = taskObject.task_classes.index(global_narrative.class_name)
        # getPlaceholderBasedSplit
        class_index = list(taskObject.classes_placeholders.keys()).index(
            global_narrative.class_name) if task_type != 'reg' else 0
        try:
            db_data = db_tables.GlobalNarration()
            db_data.task_name = global_narrative.task_name
            db_data.narrator = 45
            db_data.model_name = global_narrative.model_name
            db_data.class_name = taskObject.classes_placeholders[
                global_narrative.class_name] if task_type != 'reg' else f'{global_narrative.class_name}'
            db_data.class_label = f'C{class_index+1}' if task_type != 'reg' else f'{global_narrative.class_name}'
            db_data.features_placeholder = global_narrative.features_placeholder
            db_data.feature_division = global_narrative.feature_division
            db_data.narration = global_narrative.narration
            db_data.narrative_question = global_narrative.narrative_question
            annotator_info = getAnnotatorInfo(request)
            db_data.user_ip = annotator_info['ip']
            new_data = applyDbCommit(db, db_data)
            code = getAnnotationRewardCode('global', db, new_data.id, request)
            # db.add(db_data)
            # db.commit()
            return {'status': 'Success',
                    'Message': 'Narrative Saved',
                    'saved_data': new_data,
                    'redeem_code': code}

            # db.add(db_data)
            # db.commit()
            # return {'status': 'Success', 'Message': 'Narrative Saved'}
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
        return {'status': 'Error', 'Message': 'An error occured. Please try again later.'}
    else:
        return {'status': 'Error', 'Message': 'Invalid Model'}



@app.post('/local_narrative')
async def add_local_narrative(request: Request, local_narrative: schema.LocalNarrative, db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(
        local_narrative.task_name)
    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}

    try:
        actual_model = taskObject.modelIdx[local_narrative.model_name]
    except:
        actual_model = local_narrative.model_name
    local_narrative.model_name = actual_model
    model_details = taskObject[actual_model]
    model_details = taskObject[local_narrative.model_name]
    # current_user = checkisLogin(request, force_template=False)
    task_type = taskObject.task_type
    print(local_narrative)
    print(model_details)
    if model_details['model']:
        FCC = taskObject.modelsGlobalExplainer[local_narrative.model_name]
        if task_type != 'reg':
            class_index = list(taskObject.classes_placeholders.keys()).index(
                local_narrative.class_name)
        else:
            class_index = 0
        # class_index = taskObject.task_classes.index(local_narrative.class_name)
    try:
        db_data = db_tables.LocalNarration()
        db_data.task_name = local_narrative.task_name
        # db_data.narrator = local_narrative.narrator
        db_data.narrator = 45
        db_data.mturk_id = local_narrative.mturk_id
        db_data.model_name = local_narrative.model_name
        db_data.predicted_class = f'C{class_index+1}' if task_type != 'reg' else f'{local_narrative.class_name}'
        # db_data.class_name = taskObject.classes_placeholders[global_narrative.class_name]
        # db_data.class_label = f'C{class_index+1}'
        db_data.predicted_class_label = taskObject.classes_placeholders[
            local_narrative.class_name] if task_type != 'reg' else f'{local_narrative.class_name}'
        db_data.test_instance = local_narrative.test_instance
        db_data.feature_division = local_narrative.feature_division
        db_data.features_placeholder = local_narrative.features_placeholder
        db_data.narration = local_narrative.narration
        db_data.prediction_confidence = local_narrative.prediction_confidence
        db_data.narrative_question = local_narrative.narrative_question
        annotator_info = getAnnotatorInfo(request)
        db_data.user_ip = annotator_info['ip']
        new_data = applyDbCommit(db, db_data)
        code = getAnnotationRewardCode('local', db, new_data.id, request)
        return {'status': 'Success',
                'Message': 'Narrative Saved',
                'saved_data': new_data,
                'redeem_code': code}
    except Exception as e:
        traceback.print_exc(file=sys.stdout)
        return {'status': 'Error', 'Message': 'An error occured. Please try again later.'}


@app.post('/compare_narrative')
async def add_compare_narrative(request: Request, compare_narrative: schema.CompareNarrative, db: Session = Depends(get_db)):
    annotationInstance = AnnotationSession()
    isValid, taskObject = annotationInstance.isValidTask(
        compare_narrative.task_name)
    if not isValid:
        return {'status': 'Error', 'Message': 'Invalid Task Selected'}

    # model_details = taskObject[compare_narrative.model_name]
    try:
        actual_model = taskObject.modelIdx[compare_narrative.model_name]
    except:
        actual_model = compare_narrative.model_name

    compare_narrative.model_name = actual_model
    model_details = taskObject[actual_model]
    model_details = taskObject[compare_narrative.model_name]
    current_user = checkisLogin(request, force_template=False)
    task_type = taskObject.task_type
    if model_details['model']:
        FCC = taskObject.modelsGlobalExplainer[compare_narrative.model_name]
        class_index = list(taskObject.classes_placeholders.keys()).index(
            compare_narrative.class_name) if task_type != 'reg' else 0
        try:
            db_data = db_tables.CompareNarration()
            db_data.task_name = compare_narrative.task_name
            # current_user['id']  # compare_narrative.narrator
            db_data.narrator = 45
            db_data.model_name = compare_narrative.model_name
            db_data.predicted_class = f'C{class_index+1}' if task_type != 'reg' else f'{compare_narrative.class_name}'
            db_data.predicted_class_label = taskObject.classes_placeholders[
                compare_narrative.class_name] if task_type != 'reg' else f'{compare_narrative.class_name}'
            db_data.test_instance = compare_narrative.test_instance
            db_data.local_feature_division = compare_narrative.local_feature_division
            db_data.local_features_placeholder = compare_narrative.local_features_placeholder
            db_data.global_feature_division = compare_narrative.global_feature_division
            db_data.global_features_placeholder = compare_narrative.global_features_placeholder
            db_data.narration = compare_narrative.narration
            db_data.prediction_confidence = compare_narrative.prediction_confidence
            db_data.narrative_question = compare_narrative.narrative_question
            annotator_info = getAnnotatorInfo(request)
            db_data.user_ip = annotator_info['ip']
            new_data = applyDbCommit(db, db_data)
            code = getAnnotationRewardCode('compare', db, new_data.id, request)
            return {'status': 'Success',
                    'Message': 'Narrative Saved',
                    'saved_data': new_data,
                    'redeem_code': code}
            # return {'status': 'Success', 'Message': 'Narrative Saved'}
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            return {'status': 'Error', 'Message': 'An error occured. Please try again later.'}
    else:
        return {'status': 'Error', 'Message': 'An error occured. Please login again to retry.'}


@app.get('/help_pages/performance/{model_count_code}')
async def loadPerformanceNarrativeHelp(request: Request, model_count_code: str):
    nb_models = utils.validateURLCode(model_count_code.upper(),
                                acceptable_extension_codes)
    client_host = request.client.host
    # print(client_host)
    if nb_models is None:
        return templates.TemplateResponse('page_not_found.html', {
            'request': request, })
    examples = json.load(open('example_narratives.json'))[
        model_count_code.upper()]

    return templates.TemplateResponse('performance_narrative_help.html', {
        'request': request,
        'model_count_code': model_count_code.upper(),
        'example_narratives': examples,
        # 'vstatus': -1, 'Message': 'Session Expired. Please login again.'
    })


@app.get('/help_pages/local-level-exp')
async def loadPerformanceNarrativeHelp(request: Request, ):

    client_host = request.client.host
    examples = json.load(open('example_narratives.json'))["local_exp"]

    return templates.TemplateResponse('local_narrative_help.html', {
        'request': request,
        'example_narratives': examples,
    })
# global_narrative_help.html


@app.get('/help_pages/global-level-exp')
async def loadPerformanceNarrativeHelp(request: Request, ):
    client_host = request.client.host
    examples = json.load(open('example_narratives.json'))["global_exp"]

    return templates.TemplateResponse('global_narrative_help.html', {
        'request': request,
        'example_narratives': examples,
    })
