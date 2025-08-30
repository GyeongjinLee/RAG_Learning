# 환경 설정 (Windows)

## git 설치

 **Git은 소스코드를 안전하게 관리하고, 변경 이력을 추적할 수 있는 '버전 관리 시스템(Version Control System)'입니다.**

쉽게 말하면 **문서나 코드의 자동 저장소이자 되돌리기 버튼**이며, **여러 사람이 동시에 작업할 수 있도록 도와주는 협업 도구**입니다.

[https://git-scm.com/download/win](https://git-scm.com/download/win)

64-bit Git for Windows Setup 다운로드
Intel/AMD CPU 는 모두 X86 아키텍처를 사용하고 있습니다. 
#### Standalone Installer
**[Git for Windows/x64 Setup](https://github.com/git-for-windows/git/releases/download/v2.50.1.windows.1/Git-2.50.1-64-bit.exe).**

설치
![[images/20250726170945.png]](images/20250726170945.png)


- **설치시 옵션 체크 후 진행(윈도우 power shell(터미털)에 설치된 git 명령어 사용 할 수 있게 등록하는 옵션)**
	- **(NEW!)Add a Git Bash Profile to Windows Terminal**

![[images/Pasted image 20250727235138.png]](images/20250727235138.png)



나머지는 전부 Next 버튼을 눌러 설치를 진행합니다.


**Window 키 - PowerShell 을 반드시 **관리자 권한으로 실행**

아래의 명령어 "`git`" 을 입력하여 아래의 이미지 처럼 출력이 뜨는지 확인

```bash
git
```

결과 (대충 비슷하게 뜨면 됩니다)
* 명령어가 없다는 에러 메시지가 나오는 경우  현재 설치된 git명령를 인식 못했기 때문입니다. 이럴때는 power shell 종료 후 재 실행하면 최신 환경정보가 업데이트 되고 문제를 해결 할 수 있습니다.

![[images/20250727235436.png]](images/20250727235436.png)
## Visual Studio Code 설치

**VS Code는 마이크로소프트에서 개발한 무료 코드 편집기(코드 에디터)**입니다.  
다양한 프로그래밍 언어를 지원하고, 가볍지만 강력한 개발 도구입니다.
Visual Studio Code 다운로드

- 다운로드 링크: [https://code.visualstudio.com/download](https://code.visualstudio.com/download)

다운로드 받은 Visual Studio Code 를 설치합니다 (Applications 폴더에 복사)

Visual Studio Code 실행 후 왼쪽 install extensions 클릭
![[images/20250728001122.png]](images/20250728001122.png)

### python 검색 후 설치
* python 언어 개발 환경 지원하는 확장 팩
![[images/20250728001257.png]](images/20250728001257.png)
### jupyter 검색 후 설치
* **Jupyter는 코드, 수식, 시각화, 설명 문서를 한 화면에서 작성하고 실행할 수 있는 대화형 개발 환경**입니다.  특히 **데이터 분석, 머신러닝, 교육용 문서 작성** 등에 매우 널리 사용됩니다.
![[image/20250728001402.png]](images/20250728001402.png)
Visual Studio Code 껐다가 재실행

---  
# python 설치 및 패키지 관리자 설치치
## PowerShell Policy 적용

먼저, **Windows PowerShell** 을 **"관리자 권한으로 실행"** 합니다.
![[images/20250726171358.png]](images/20250726171358.png)

다음의 명령어를 입력하여 Policy 를 적용합니다.

```bash

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

적용이 완료된 후 **Windows PowerShell** 을 껐다가 켭니다. 아래의 진행을 위하여 **Windows PowerShell** 실행시 **"관리자 권한으로 실행"** 합니다.

이 명령어는 **현재 사용자 계정에 한해 PowerShell 스크립트의 실행을 허용**하되, **인터넷에서 다운로드한 스크립트는 디지털 서명이 필요하도록 설정**합니다.  
보안과 유연성을 절충한 설정으로, 로컬 개발이나 테스트에 자주 사용됩니다.

## pyenv 설치
#### 목적
- `pyenv`는 "여러 Python 버전을 안전하고 유연하게 관리하기 위한 도구"이며, 프로젝트별 맞춤형 개발 환경을 구성할 수 있게 도와줍니다.

| 구성 요소                                        | 설명                                                                                                                   |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `git clone <URL>`                            | 해당 Git 저장소를 클론(복제)합니다.                                                                                               |
| `https://github.com/pyenv-win/pyenv-win.git` | [pyenv-win](https://github.com/pyenv-win/pyenv-win) 프로젝트의 GitHub 저장소 URL입니다. Windows에서 Python 버전 관리를 가능하게 해주는 도구입니다. |
| `"$env:USERPROFILE\.pyenv"`                  | 현재 사용자 계정의 홈 디렉터리에 `.pyenv` 폴더로 복제합니다. 예: `C:\Users\사용자이름\.pyenv`                                                    |
```bash
git clone https://github.com/pyenv-win/pyenv-win.git "$env:USERPROFILE\.pyenv"
```

**환경변수 추가**

아래의 내용을 복사하여 붙혀넣기 후 실행

```bash
[System.Environment]::SetEnvironmentVariable('PYENV', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_ROOT', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
[System.Environment]::SetEnvironmentVariable('PYENV_HOME', $env:USERPROFILE + "\.pyenv\pyenv-win\", "User")
```

**구성 요소별 설명**

| 항목                                        | 설명                                                             |
| ----------------------------------------- | -------------------------------------------------------------- |
| `SetEnvironmentVariable`                  | 윈도우 사용자 환경변수를 설정합니다.                                           |
| `'PYENV'`, `'PYENV_ROOT'`, `'PYENV_HOME'` | `pyenv-win`에서 사용하는 환경 변수명입니다.                                  |
| `$env:USERPROFILE + "\.pyenv\pyenv-win\"` | 현재 사용자 폴더 기준 설치 경로를 설정합니다.예: `C:\Users\사용자명\.pyenv\pyenv-win\` |
| `"User"`                                  | 해당 설정이 현재 사용자 계정 수준에서만 적용되도록 지정합니다.                            |
아래의 내용을 복사하여 붙혀넣기 후 실행

```bash
[System.Environment]::SetEnvironmentVariable('PATH', $env:USERPROFILE + "\.pyenv\pyenv-win\bin;" + $env:USERPROFILE + "\.pyenv\pyenv-win\shims;" + [System.Environment]::GetEnvironmentVariable('PATH', "User"), "User")
```

**구성 요소별 설명**

| 요소                                                               | 설명                                            |
| ---------------------------------------------------------------- | --------------------------------------------- |
| `'PATH'`                                                         | 시스템에서 실행 파일을 찾는 데 사용되는 **환경 변수**입니다.          |
| `$env:USERPROFILE + "\.pyenv\pyenv-win\bin"`                     | pyenv의 실행 파일(`pyenv.exe` 등)이 위치한 경로입니다.       |
| `$env:USERPROFILE + "\.pyenv\pyenv-win\shims"`                   | Python 실행 파일(`python.exe` 등)의 **shim** 경로입니다. |
| `+ [System.Environment]::GetEnvironmentVariable('PATH', 'User')` | 기존 사용자 PATH를 가져와 추가 경로를 **보존**합니다.            |
| `"User"`                                                         | 현재 사용자 계정에 대해 설정합니다 (전역 시스템이 아님).             |
현재의 **Windows PowerShell** 을 종료 후 다시 실행합니다.

다음의 명령어를 입력하여 정상 동작하는지 확인합니다.

```bash
pyenv
```

# 실습코드 다운로드

- RAG 실습코드 링크: https://github.com/GyeongjinLee/RAG_Learning.git
* 사용자 모드로  power shell 실행

![[images/20250726174732.png]](images/20250726174732.png)

![[images/20250726214449.png]](images/20250726214449.png)

1. 사용자의 Documents 디렉토리로 변경  
```bash
cd $HOME/Documents
pwd
```

2. 아래의 명령어를 실행하여 소스코드를 받습니다.
github repo에 저장된 자료는 강의 샘플코드 개발환경 설정에 대한 정보는 VSCODE를 실행하여 설정 합니다.

```bash
git clone https://github.com/GyeongjinLee/RAG_Learning.git
```

생성된 RAG_Learning 디렉토리로 이동
```bash
cd $HOME/Documents/RAG_Learning
```

## python 설치

파이썬 3.11 버전 설치
## ✅ 명령어 의미

- 이 명령은 **Python 3.11.9 버전을 설치**하는 명령입니다.  
- `pyenv`를 통해 설치되므로, 시스템 Python에 영향을 주지 않고 **독립적으로** Python을 설치하고 사용할 수 있습니다.
-
```bash
cd $HOME/Documents/RAG_Learning     #git에서 복제한 실습 코드가 있는 디렉토리 이동 
pyenv install 3.11.9
```

#### 설치 확인 명렁어 리스트

```bash
pyenv versions         # 설치된 python 버전 목록 확인하는 명령어
pyenv global 3.11.9    # 모든 사용자가 사용하는 기본 Python 버전 설정
python --version       # 현재 적용 중인 버전 확인
```

**로컬(Local)버전 변경**
* 현재 디렉터리(예: 프로젝트 폴더)에 `.python-version` 파일을 생성하고,  
    해당 위치에서만 적용될 **로컬 Python 버전**을 설정합니다.
- 프로젝트마다 다른 Python 버전을 요구할 때 유용합니다.
    
```bash
cd $HOME/Documents/RAG_Learning     #git에서 복제한 실습 코드가 있는 디렉토리 이동 
pyenv local 3.11.9
python --version  # → Python 3.11.9 (myproject 내에서만)

```

## Poetry 설치

- `**Python 3 환경에서 Poetry 도구를 1.8.5 버전으로 설치**하여, **의존성 및 프로젝트 관리를 자동화**할 수 있게 만드는 명령어입니다.
- 아래의 명령어를 실행하여 Poetry 패키지 관리 도구를 설치합니다.

```bash
curl.exe -sSL https://install.python-poetry.org | python -
```

poetry 를 이용하여 파이썬 가상환경 및 패키지 관리를 위한 초기화
* 실습코드가 있는 디렉토리에서 실행
* 실행 완료 후  pyproject.toml 파일 생성됨
```bash
poetry init --name "RAG_Learning" --description "RAG 학습" --python "^3.11.9" --no-interaction
```
```bash
참고
poetry 설치 위치는 보통 아래와 같습니다.
C:\Users\<사용자명>\AppData\Roaming\Python\Scripts

$oldPath = [Environment]::GetEnvironmentVariable("Path", "User")
$newPath = "$oldPath;C:\Users\LeeGyeongjin\AppData\Roaming\Python\Scripts"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")
```

파이썬 패키지 일괄 업데이트

```bash
#requirments.txt 의 패키지 정보를 기반으로 설치 및 업데이트를 수행함
poetry add $(cat requirements.txt)
```


**설치가 다 되었으면, VS Code를 실행 상단 메뉴 File -> Open Folder 실습코드가 있는 디렉토리를 오픈**

- 우측 상단 "select kernel"

![[images/20250727183819.png]](images/20250727183819.png)
python environment 클릭 - 3.11.9 가상 환경 선택
- 설치한 가상환경이 안뜬다면 Visual Studio Code 껐다가 재실행
![[images/20250727190655.png]](images/20250727190655.png)
