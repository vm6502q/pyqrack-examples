@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Loop over qubit width from 4 to 28
FOR %%W IN (4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28) DO (
    :: Calculate 1/4 of qubit width, rounded up
    SET /A QUARTER_WIDTH=%%W / 4
    SET /A REMAINDER=%%W %% 4
    IF !REMAINDER! GTR 0 SET /A QUARTER_WIDTH+=1

    :: Set environment variables for this width
    SET QRACK_MAX_PAGING_QB=!QUARTER_WIDTH!
    SET QRACK_MAX_CPU_QB=!QUARTER_WIDTH!

    python sycamore_2019_qiskit_validation.py %%W %%W
)

ENDLOCAL
