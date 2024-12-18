FOR /L %%i IN (28,1,30) DO (
  FOR /L %%j IN (1,1,%%i) DO (
    call python sycamore_2019_patch_quadrant.py %%i %%j
  )
)
