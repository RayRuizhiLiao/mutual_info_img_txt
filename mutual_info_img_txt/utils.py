class MimicID:
	subject_id = ''
	study_id = ''
	dicom_id = ''

	def __init__(self, subject_id, study_id, dicom_id):
		self.subject_id = str(subject_id)
		self.study_id = str(study_id)
		self.dicom_id = str(dicom_id)

	def __str__(self):
		return f"p{self.subject_id}_s{self.study_id}_{self.dicom_id}"

	@staticmethod
	def get_study_id(mimic_id: str):
		return mimic_id.split('_')[1][1:]
