# Models predictions

## predictClass
Принимает путь к изображению и предсказывает к какому классу оно относится, возвращает строку

Классы, предсказания:
['jaw-lower',
'jaw-upper',
'mouth-sagittal_fissure',
'mouth-vestibule-front-closed',
'mouth-vestibule-front-half_open',
'mouth-vestibule-half_profile-closed-left',
'mouth-vestibule-half_profile-closed-right',
'mouth-vestibule-profile-closed-left',
'mouth-vestibule-profile-closed-right',
'portrait']

## predictLandmarksAndAngle 
Принимает путь к портретному изображению, предсказываются угол поворота головы, и лендмарки лица. В случае если голова повернута больше чем на 50 градусов (профиль) предсказывается 39 точек, если анфас то 468. Возвращается кортеж: двумерный массив numpy с координатами точек, угол поворота лица.

## getFaceBox (пример)
Принимает ширину и высоту изображения, массив координат точек лица, угол поворота лица. Возвращает кортеж подсчитанные координаты границы лица для обрезания (left, right, top, bot), угол поворота для выравнивания