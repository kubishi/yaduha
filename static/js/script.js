const dropdownIDLabels = {
    '#subject-noun': 'Subject',
    '#subject-suffix': 'Suffix',
    '#object-noun': 'Object',
    '#object-suffix': 'Suffix',
    '#object-pronoun': 'Object Pronoun',
    '#verb': 'Verb',
    '#verb-tense': 'Tense'
}

const dropdowns = {};
Object.keys(dropdownIDLabels).forEach((id, index) => {
    dropdowns[id] = new Choices(id, {
        itemSelectText: '',
        removeItems: true,
        removeItemButton: true,
        placeholder: true,
        placeholderValue: '',
        shouldSort: false,
    });
});

function setChoices(dropdownID, choices, value, requirement) {
    $(dropdownID).empty()
    // var element = document.getElementById(dropdownID);
    const choicesDropdown = dropdowns[dropdownID];

    if (choices.length == 0) {
        choicesDropdown.setChoices([{
            value: '',
            label: dropdownIDLabels[dropdownID]
        }], 'value', 'label', true);
    } else {
        // choicesDropdown.setChoices(choices.map(choice => {
        //     return {
        //         value: choice[0],
        //         label: choice[1]
        //     }
        // }), 'value', 'label', true);
        // include the placeholder
        choicesDropdown.setChoices([{
            value: '',
            label: dropdownIDLabels[dropdownID]
        }].concat(choices.map(choice => {
            return {
                value: choice[0],
                label: choice[1]
            }
        }
        )), 'value', 'label', true);
    }

    // get label for value from choices
    var label = value;
    for (var i = 0; i < choices.length; i++) {
        if (choices[i][0] == value) {
            label = choices[i][1]
            break
        }
    }

    // set the value
    if (value != null) {
        // choicesDropdown.setValue([{value: value, label: label}])
        choicesDropdown.setChoiceByValue(value)
    } else {
        // choicesDropdown.setValue([{value: null, label: dropdownIDLabels[dropdownID]}])
        choicesDropdown.setChoiceByValue('')
    }

    // set the requirement
    if (requirement == "required") {
        choicesDropdown.enable()
    } else if (requirement == "optional") {
        choicesDropdown.enable()
    } else if (requirement == "disabled") {
        choicesDropdown.disable()
    }
}

function updateDropdowns(url) {
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": $('#subject-noun').val(),
            "subject_suffix": $('#subject-suffix').val(),
            "verb": $('#verb').val(),
            "verb_tense": $('#verb-tense').val(),
            "object_pronoun": $('#object-pronoun').val(),
            "object_noun": $('#object-noun').val(),
            "object_suffix": $('#object-suffix').val()
        })
    }).catch(err => {
        console.error(err);
    }).then(response => response.json()).then(res => {
        choices = res.choices
        setChoices(
            '#subject-noun', 
            choices.subject_noun.choices, choices.subject_noun.value, choices.subject_noun.requirement
        )
        setChoices(
            '#subject-suffix',
            choices.subject_suffix.choices, choices.subject_suffix.value, choices.subject_suffix.requirement
        )
        setChoices(
            '#verb', 
            choices.verb.choices, choices.verb.value, choices.verb.requirement
        )
        setChoices(
            '#verb-tense', 
            choices.verb_tense.choices, choices.verb_tense.value, choices.verb_tense.requirement
        )
        setChoices(
            '#object-pronoun', 
            choices.object_pronoun.choices, choices.object_pronoun.value, choices.object_pronoun.requirement
        )
        setChoices(
            '#object-noun', 
            choices.object_noun.choices, choices.object_noun.value, choices.object_noun.requirement
        )
        setChoices(
            '#object-suffix', 
            choices.object_suffix.choices, choices.object_suffix.value, choices.object_suffix.requirement
        )

        // Update sentence if it exists
        if (res.sentence.length > 0) {
            formattedSentence = res.sentence.map(word => {
                if (word.type == "subject") {
                    return `<span class="text-danger">${word.text}</span>`
                } else if (word.type == "verb") {
                    return `<span class="text-primary">${word.text}</span>`
                } else if (word.type == "object") {
                    return `<span class="text-success">${word.text}</span>`
                } else {
                    return word.text
                }
            })

            // Set contents of #sentence to formattedSentence
            $('#sentence').html(formattedSentence.join(" "))
            // make btn-translate visible by removing display: none
            $('#btn-translate').css('display', '')
            // enable btn-translate
            $('#btn-translate').prop('disabled', false)
            // remove translation 
            $('#translation').html("")
            // set btn-randomize to invisible
            $('#btn-randomize').css('display', 'none')
        } else {
            $('#sentence').html("Your sentence isn't valid yet, please select more words.")
            // make btn-translate invisible
            $('#btn-translate').css('display', 'none')
            // remove translation 
            $('#translation').html("")
            // set btn-random to default display
            $('#btn-randomize').css('display', '')
        }
    });
}

// implement translate button
$('#btn-translate').click(function() {
    // get the current values of all dropdowns
    subject_noun = $('#subject-noun').val()
    subject_suffix = $('#subject-suffix').val()
    verb = $('#verb').val()
    verb_tense = $('#verb-tense').val()
    object_pronoun = $('#object-pronoun').val()
    object_noun = $('#object-noun').val()
    object_suffix = $('#object-suffix').val()

    // disable btn-translate
    $('#btn-translate').prop('disabled', true)

    // call /api/translate with the current values of all dropdowns
    fetch('/api/translate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            "subject_noun": subject_noun,
            "subject_suffix": subject_suffix,
            "verb": verb,
            "verb_tense": verb_tense,
            "object_pronoun": object_pronoun,
            "object_noun": object_noun,
            "object_suffix": object_suffix
        })
    }).catch(err => {
        console.error(err);
        // enable btn-translate
        $('#btn-translate').prop('disabled', false)
    }).then(response => response.json()).then(res => {
        // set the contents of #translation to the returned translation
        $('#translation').html(res.translation)
        // make btn-translate invisible
        $('#btn-translate').css('display', 'none')
    });
})

$(function () {
    $('[data-toggle="help-subject-popover"]').popover()
    $('[data-toggle="help-verb-popover"]').popover()
    $('[data-toggle="help-object-popover"]').popover()
})

$('body').on('click', function (e) {
    //did not click a popover toggle or popover
    if (!$(e.target).hasClass('popover') && !$(e.target).hasClass('popover-header') && !$(e.target).hasClass('popover-body')) {
        $('#help-subject-popover').popover('hide');
        $('#help-verb-popover').popover('hide');
        $('#help-object-popover').popover('hide');
    }
});

// call updateDropdowns() when the page loads
$(document).ready(function() {
    $('#help-subject-popover').popover({
        title: 'What is a subject?',
        content: $('#help-subject').html(),
        html: true
    });

    $('#help-verb-popover').popover({
        title: 'What is a verb?',
        content: $('#help-verb').html(),
        html: true
    });

    $('#help-object-popover').popover({
        title: 'What is an object?',
        content: $('#help-object').html(),
        html: true
    });

    updateDropdowns('/api/choices')
})
