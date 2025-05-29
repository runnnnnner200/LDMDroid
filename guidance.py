all_operation_workflows = {
    "C": [
        "Open the create interface",
        "Choose the creation type",
        "Enter the necessary information",
        "Submit and save",
        "Return to the list",
    ],
    "R": [
        "Select the item to read",
        "Open the detail interface",
        "Read the detail",
    ],
    "U": [
        "Select the item to edit",
        "open the edit interface",
        "Enter the modification information",
        "Submit and save",
        "Return to the list",
    ],
    "D": [
        "Select the item to delete",
        "Delete the item",
        "Confirm the deletion",
        "Return to the list",
    ],
    "S": [
        "Open the search interface",
        "Enter the search keyword",
        "Trigger the search action",
        "Read the search result",
    ]
}

all_operation_guidance = {
    "C": [
        "Some apps may not have an explicit “Submit and save” step. In these cases, the app might automatically "
        "save the information once it’s entered. If you cannot find a save button, try using the back button or "
        "sending a “back” action",
    ],
    "R": [],
    "U": [
        "Some apps may not have an explicit “Submit and save” step. In these cases, the app might automatically "
        "save the information once it’s entered. If you cannot find a save button, try using the back button or "
        "sending a “back” action"
    ],
    "D": [],
    "S": [
        "In some apps, the search action may be triggered automatically without the “Trigger the search action” step."]
}

all_determine_stage_guidance = {
    "C": [
        "Some steps may be skipped or completed implicitly depending on the app's behavior.",
        "Some apps may not have an explicit “Submit and save” step. In these cases, the app might automatically "
        "save the information once it’s entered."
    ],
    "R": [
        "Some steps may be skipped or completed implicitly depending on the app's behavior."
    ],
    "U": [
        "Sometimes once you select the item to edit, you will open the edit interface at the same time",
        "Some steps may be skipped or completed implicitly depending on the app's behavior.",
    ],
    "D": [
        "Some steps may be skipped or completed implicitly depending on the app's behavior."
    ],
    "S": [
        "Some steps may be skipped or completed implicitly depending on the app's behavior.",
        "In some apps, the search action may be triggered automatically without the “Trigger the search action” step."
    ]
}

all_test_oracle_guidance = {
    "C": "Create: Detect if there is any logical error in the Create operation. A logical error occurs if the target "
         "data was not correctly added to the data list. Focus on whether the target data appears in the data list "
         "after the operation.",
    "R": "Read: Detect if there is any logical error in the Read operation. A logical error occurs if the target data "
         "was not correctly fetched or displayed as expected. Focus on whether the target data's details are "
         "correctly displayed, and ensure the right item is being viewed.",
    "U": "Update: Detect if there is any logical error in the Update operation. A logical error occurs if the target "
         "data was not correctly modified as expected. Focus on whether the target data has been updated with the "
         "correct values in the data list.\n\n"
         "The position of the target data in the data list may change after the Update operation, as the app may sort "
         "the list based on certain criteria",
    "D": "Detect if there is any logical error in the Delete operation. A logical error occurs if the target data was "
         "not correctly removed. Focus on whether the target data is still present or correctly removed in the data "
         "list.",
    "S": "Search: Detect if there is any logical error in the Search operation. A logical error occurs if the target data "
         "does not appear in the search results as expected. Focus on whether the target data appears in the results",
    "Additional": [
        "The details shown in the data list could be truncated or omitted, because the data list interface cannot "
        "display all information.",
        "Ignore minor differences in timestamps UI elements, because they may change automatically.",
        "Each data operation will only operate on one target data."
    ]
}
